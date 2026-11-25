FROM python:3.12-slim
WORKDIR /app
COPY jais-chat.html /app/index.html
COPY server.py /app/server.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
