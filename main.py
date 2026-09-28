"""CLI：导入 PDF/TXT、构建或复用向量库、两轮演示与持续问答。"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from rag_engine import ROOT, RAGEngine, create_embedding, create_llm, import_file

DEMO = ["公司出差的住宿报销上限是多少？", "那餐饮补贴呢？"]


def show_sources(sources):
    for source in sources:
        print(f"\n[{source.number}] {source.filename} · 第 {source.page} 页/文本段 · 相似度 {source.score:.3f}")
        print(source.text)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="LlamaIndex + Chroma 中文 RAG 知识库")
    parser.add_argument("--import", dest="imports", action="append", type=Path, default=[], help="导入PDF/TXT，可重复指定")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--storage-dir", type=Path, default=ROOT / "storage")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--chat", action="store_true", help="跳过两轮演示")
    modes.add_argument("--demo-only", action="store_true", help="演示后退出")
    modes.add_argument("--ingest-only", action="store_true", help="只解析并入库，无需LLM密钥")
    modes.add_argument("--retrieve-only", metavar="QUESTION", help="只运行真实向量检索，无需LLM密钥")
    args = parser.parse_args()
    try:
        llm = None if args.ingest_only or args.retrieve_only else create_llm()
        for path in args.imports:
            print(f"已导入：{import_file(path, args.data_dir)}")
        print("正在加载本地 Embedding 模型；首次运行需下载模型，请稍候。", flush=True)
        embedding = create_embedding(args.storage_dir / "models")
        engine = RAGEngine(args.data_dir, args.storage_dir / "chroma", embedding, llm,
                           min_score=float(os.getenv("MIN_SCORE", "0.45")))
        print(f"知识库就绪：{engine.chunk_count} 个片段，{'复用已有向量' if engine.reused else '完成向量化入库'}。")
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, UnicodeError)):
            print(f"初始化失败：{exc}", file=sys.stderr)
        else:
            print(f"初始化失败（{type(exc).__name__}）：检查依赖、模型下载网络、PDF和目录权限。", file=sys.stderr)
        return 2
    if args.ingest_only:
        return 0
    if args.retrieve_only:
        try:
            sources = engine.retrieve(args.retrieve_only)
            show_sources(sources)
            if not sources:
                print("未找到相关片段。")
            return 0
        except Exception as exc:
            print(f"检索失败（{type(exc).__name__}）。", file=sys.stderr)
            return 1

    def ask(question):
        print(f"\n用户：{question}")
        try:
            answer = engine.ask(question)
            print(f"检索问题：{answer.query}\n助手：{answer.text}")
            show_sources(answer.sources)
            return True
        except Exception as exc:
            print(f"问答失败（{type(exc).__name__}）：请检查API密钥、额度、网络或重试；本轮未写入记忆。", file=sys.stderr)
            return False

    if not args.chat:
        for question in DEMO:
            if not ask(question):
                return 1
    if args.demo_only or not sys.stdin.isatty():
        return 0
    print("输入问题继续对话，/reset 清空记忆，/exit 退出。")
    while True:
        try:
            text = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见。")
            return 0
        if text == "/exit":
            return 0
        if text == "/reset":
            engine.reset()
            print("对话记忆已清空，向量库仍保留。")
        elif text:
            ask(text)


if __name__ == "__main__":
    raise SystemExit(main())
