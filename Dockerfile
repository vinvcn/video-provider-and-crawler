# vpc-retrieval —— Phase 2 检索服务的运行镜像。
# 构建上下文 = 仓库根目录；.dockerignore 把 storage/（68GB 缩略图）与 .env（密钥）
# 挡在镜像之外。运行时配置（DSN、嵌入 API key）由 compose 的 env_file 注入。
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY crawler/ crawler/
COPY store/ store/
COPY service/ service/
COPY bench/ bench/

RUN pip install --no-cache-dir ".[service]"

EXPOSE 8787

# /healthz 走本进程：200 即服务可用（dense/sparse 覆盖率在响应体里）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=4)"]

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8787"]
