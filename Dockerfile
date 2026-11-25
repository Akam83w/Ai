FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY model.py tokenizer.py api_server.py jais-chat.html /app/
RUN mkdir -p /app/checkpoints
EXPOSE 8080
CMD ["python", "api_server.py"]
