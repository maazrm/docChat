FROM python:3.11-slim

WORKDIR /app

# System dependencies for docling (OpenCV + image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/chroma_db uploads

# Pre-download docling layout models (~1-2 GB) into the image cache
# Remove this line if you prefer to download on first run via a mounted volume
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()" 2>/dev/null || true

EXPOSE 8501

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]
