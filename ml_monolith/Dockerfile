FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application code
COPY . .

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Dummy entrypoint - will be overridden in docker-compose
CMD ["python", "--version"]
