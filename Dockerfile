# Official Python image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src

# Default working directory when container runs
WORKDIR /app/src

# Run shell and keep container alive
CMD ["bash"]
