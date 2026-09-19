# 最小部署镜像（CLI 与服务化 API 通用）。
# 构建：docker build -t ecom-service-agent .
# 运行 API：docker run -p 8000:8000 --env-file .env ecom-service-agent python -m app.api.server
FROM python:3.12-slim

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 默认跑 CLI；如需 API，覆盖 CMD：python -m app.api.server
CMD ["python", "main.py"]