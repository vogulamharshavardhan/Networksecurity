
FROM python:3.13-slim

WORKDIR /app

# Prevent Python from generating bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and AWS CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Application port
EXPOSE 8080

# Start application
CMD ["python", "app.py"]
