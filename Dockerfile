FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Indexes and datasets live on a volume, not in the image.
VOLUME ["/data"]
ENV MINISEARCH_INDEX=/data/indexes/wiki
EXPOSE 8000

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["sh", "-c", "minisearch serve --index \"$MINISEARCH_INDEX\" --host 0.0.0.0 --port 8000"]
