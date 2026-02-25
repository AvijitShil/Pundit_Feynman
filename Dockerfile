# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Create jobs directory for temporary storage
RUN mkdir -p /app/jobs && chmod 777 /app/jobs

# Set PYTHONPATH to current directory
ENV PYTHONPATH=/app

# Expose port
EXPOSE 7860

# Command to run uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
