FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure directory structure for local paths in image_server.py
RUN mkdir -p /app/trained_models

EXPOSE 7860

CMD ["python", "image_server.py"]
