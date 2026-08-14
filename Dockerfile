# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies pip can't install:
# - tesseract-ocr + tesseract-ocr-hin: OCR fallback for scanned PDFs,
#   with Hindi support (bilingual government documents - see
#   app/ingestion/loader.py for why English-only OCR silently mangles
#   Hindi text instead of failing loudly).
# - poppler-utils: backs pdf2image's page rasterization for OCR, and
#   pdfplumber's underlying PDF parsing.
# - libpq-dev, gcc: needed to build psycopg from source if a prebuilt
#   wheel isn't available for this exact base image's platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first, separately from copying the rest of the
# code - Docker layer caching means `docker build` skips reinstalling
# ~1GB of torch/transformers on every rebuild if requirements.txt
# hasn't changed, only if the actual application code changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

# data/ is intentionally NOT copied into the image - PDFs are large,
# scheme-specific, and change independently of the code. Mount them
# as a volume at runtime instead (see docker-compose.yml). Baking
# ~50MB+ of PDFs into every image build is wasteful and makes the
# image itself carry data it doesn't need to run.

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
