FROM python:3.13-slim

WORKDIR /app

# Install build tools for C++ extensions (chroma-hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for runtime data
RUN mkdir -p uploaded_pdfs chroma_db logs

# Disable ChromaDB telemetry
ENV ANONYMIZED_TELEMETRY=False

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
