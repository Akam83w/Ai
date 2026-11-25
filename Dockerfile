# Build the React/Vite interface.
FROM node:22-alpine AS web-build
WORKDIR /app
COPY package.json ./
RUN npm install --no-package-lock
COPY . .
RUN npm run build

# Run OwnedLM with the locally loaded pretrained Arabic model.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080 \
    HF_HOME=/data/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY --from=web-build /app/dist ./dist
COPY . .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && mkdir -p /data/huggingface
EXPOSE 8080
CMD ["python", "server.py"]
