FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY main.py .
COPY prometheus.yml .

# Expose the port
EXPOSE 8080

# RUN python main.py
CMD ["python", "src/api/main.py"]