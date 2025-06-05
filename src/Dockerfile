# Use a PyTorch base image that supports CPU fallback
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy only requirements file first to leverage Docker layer caching
COPY src/requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ src/

# Set default command to bash for interactive use
CMD ["bash"]
