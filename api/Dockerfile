FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy API requirements
COPY api/requirements.txt .

# Upgrade pip first to avoid some compatibility issues
RUN pip install --upgrade pip

# Install dependencies from requirements.txt with no cache and force reinstall
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy API and src code
COPY api /app/api

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI with uvicorn
CMD ["uvicorn", "api.core.main:app", "--host", "0.0.0.0", "--port", "8000"]