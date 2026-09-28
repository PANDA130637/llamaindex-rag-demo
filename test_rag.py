"""无需API或模型下载：确定性测试向量 + 模拟LLM，实际运行LlamaIndex和Chroma。"""

import hashlib
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.base.embeddings.base import BaseEmbedding
from rag_engine import ROOT, RAGEngine, create_llm, import_file, load_documents


class TestEmbedding(BaseEmbedding):
    """测试专用字符片段哈希，不是语义模型，不用于正式演示。"""
    model_name: str = "offline-test-hash-128"

    def _get_text_embedding(self, text):
        vector = [0.0] * 128
        for i in range(max(0, len(text) - 1)):
            slot = int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % 128
            vector[slot] += 1
        norm = math.sqrt(sum(x * x for x in vector)) or 1
        return [x / norm for x in vector]

    def _get_query_embedding(self, query):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.base = Path(cls.tmp.name)
        cls.data = cls.base / "data"
        shutil.copytree(ROOT / "data", cls.data, ignore=shutil.ignore_patterns("uploads"))
        cls.embedding = TestEmbedding()
        cls.engine = RAGEngine(cls.data, cls.base / "db", cls.embedding, min_score=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.engine.reset()

    def test_txt_and_pdf_parsing(self):
        docs = load_documents(self.data)
        self.assertTrue(any(d.metadata["source"].endswith(".txt") and "300" in d.text for d in docs))
        pdf_docs = [d for d in docs if d.metadata["source"].endswith(".pdf")]
        self.assertTrue(pdf_docs)
        self.assertIn("Security", pdf_docs[0].text)
        self.assertEqual(pdf_docs[0].metadata["page"], 1)

    def test_real_chroma_retrieval_and_metadata(self):
        hits = self.engine.retrieve("住宿报销规则 每晚最高300元")
        self.assertTrue(hits)
        self.assertTrue(any("300" in s.text and s.filename == "travel_policy.txt" for s in hits))
        self.assertTrue(all(s.page >= 1 for s in hits))

    def test_persistent_reopen_does_not_duplicate(self):
        before = self.engine.collection.count()
        reopened = RAGEngine(self.data, self.base / "db", self.embedding, min_score=0)
        self.assertTrue(reopened.reused)
        self.assertEqual(reopened.collection.count(), before)
        self.assertTrue(reopened.retrieve("住宿"))

    def test_changed_document_uses_new_snapshot(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            data = Path(name)
            (data / "policy.txt").write_text("住宿上限300元", encoding="utf-8")
            old = RAGEngine(data, self.base / "change-db", self.embedding, min_score=0)
            (data / "policy.txt").write_text("住宿上限500元", encoding="utf-8")
            new = RAGEngine(data, self.base / "change-db", self.embedding, min_score=0)
            self.assertNotEqual(old.collection.name, new.collection.name)
            self.assertIn("500", new.retrieve("住宿")[0].text)

    def test_followup_rewrite_and_citations(self):
        model = ScriptedLLM(['{"answer":"住宿上限300元 [1]","citations":[1]}',
            "公司出差餐饮补贴每天多少元？", '{"answer":"餐补每天80元 [1]","citations":[1]}'])
        self.engine.llm = model
        first = self.engine.ask("住宿上限是多少？")
        second = self.engine.ask("那餐补呢？")
        self.assertEqual(len(first.sources), 1)
        self.assertEqual(second.query, "公司出差餐饮补贴每天多少元？")
        self.assertIn("住宿上限是多少", model.prompts[1])
        self.assertEqual(len(self.engine.history), 2)
        self.engine.reset()
        self.assertEqual(self.engine.history, [])

    def test_invalid_citation_rejected_without_memory(self):
        for reply in ['{"answer":"错误 [99]","citations":[99]}',
                      '{"answer":"错误 [2]","citations":[1]}', 'not-json']:
            self.engine.llm = ScriptedLLM([reply])
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                self.engine.ask("住宿")
            self.assertEqual(self.engine.history, [])

    def test_unsupported_answer_without_citations_is_refused(self):
        self.engine.llm = ScriptedLLM(['{"answer":"随便编造的结论","citations":[]}'])
        result = self.engine.ask("宠物补贴")
        self.assertIn("不足以支持", result.text)
        self.assertEqual(result.sources, [])

    def test_no_hits_does_not_call_llm(self):
        self.engine.llm = ScriptedLLM([])
        with patch.object(self.engine, "retrieve", return_value=[]):
            result = self.engine.ask("未知")
        self.assertIn("没有找到", result.text)

    def test_api_failure_does_not_change_memory(self):
        self.engine.llm = ScriptedLLM([RuntimeError("模拟API失败")])
        with self.assertRaises(RuntimeError):
            self.engine.ask("住宿")
        self.assertEqual(self.engine.history, [])

    def test_memory_keeps_six_turns(self):
        self.engine.llm = ScriptedLLM(["独立问题"] * 10)
        with patch.object(self.engine, "retrieve", return_value=[]):
            for i in range(9):
                self.engine.ask(f"问题{i}")
        self.assertEqual(len(self.engine.history), 6)
        self.assertEqual(self.engine.history[0][0], "问题3")

    def test_import_idempotence_and_collision(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            source = base / "new.txt"
            source.write_text("第一版", encoding="utf-8")
            dest = import_file(source, base / "data")
            self.assertEqual(import_file(source, base / "data"), dest)
            source.write_text("第二版", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_file(source, base / "data")

    def test_empty_and_unreadable_documents(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            with self.assertRaises(ValueError):
                load_documents(base)
            (base / "empty.txt").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_documents(base)


class ConfigurationTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            create_llm()

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-test-placeholder"}, clear=True)
    def test_llm_adapter_configuration(self):
        llm = create_llm()
        self.assertEqual(llm.model, "deepseek-flash")
        self.assertTrue(llm.is_chat_model)
        self.assertEqual(llm.additional_kwargs["extra_body"]["thinking"]["type"], "disabled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
