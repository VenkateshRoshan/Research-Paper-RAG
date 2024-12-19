# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY main.py .

# create the json file from an env variable
ARG GOOGLE_APPLICATION_CREDENTIALS
RUN echo '$GOOGLE_APPLICATION_CREDENTIALS' > /app/credentials.json

# Set environment variables for Vertex AI
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HEALTH_ROUTE=/health
ENV PORT=8080

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE ${PORT}

# Start the FastAPI server
CMD ["python3", "main.py"]