# Official Python image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy source code
COPY src/ src/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run shell and keep container alive
CMD ["bash"]
