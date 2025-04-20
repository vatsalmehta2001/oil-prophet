FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyTorch and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OIL_PROPHET_API_KEY=your_api_key_here

# Create directories if they don't exist
RUN mkdir -p models data/processed/reddit_test_small notebooks/plots cache

# Expose port for API
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]