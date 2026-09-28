FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home agent && mkdir -p /app/storage && chown agent:agent /app/storage
COPY --chown=agent:agent main.py rag_engine.py test_rag.py ./
COPY --chown=agent:agent data ./data
USER agent
VOLUME ["/app/storage"]
CMD ["python", "main.py"]
