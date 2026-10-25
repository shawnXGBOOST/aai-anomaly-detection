FROM python:3.11-slim

ARG WITH_TORCH=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AAI_ARTIFACTS=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# CPU-only PyTorch is only needed to serve the autoencoder: --build-arg WITH_TORCH=1
RUN pip install --no-cache-dir ".[api]" \
    && if [ "$WITH_TORCH" = "1" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
       fi

RUN useradd --create-home --uid 10001 aai
USER aai

# Artifacts are mounted, not baked in:  docker run -v $PWD/artifacts/models:/models:ro ...
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "--factory", "aai.serving.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
