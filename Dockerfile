FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY dpo/ ./dpo/

# Default: smoke-test run (override CMD in Cloud Build / Cloud Run as needed)
CMD ["python", "-m", "dpo.train_trl", "--dataset_name", "hh", "--output_dir", "/output"]
