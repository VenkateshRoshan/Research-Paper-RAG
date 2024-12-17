FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY src/ src/
COPY tests/ tests/
COPY main.py .
COPY prometheus.yml .

# RUN python main.py
CMD ["python", "main.py"]