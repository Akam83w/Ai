# Build the React/Vite interface.
FROM node:22-alpine AS web-build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Run the local Python model API and serve the built interface from one port.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080
WORKDIR /app
COPY --from=web-build /app/dist ./dist
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8080
CMD ["python", "server.py"]
