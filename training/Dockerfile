FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts and src for model.py
COPY training/ /app/training/
COPY src/ /app/src/

# Entrypoint
CMD ["python", "training/main.py"]