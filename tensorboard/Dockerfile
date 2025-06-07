FROM python:3.10-slim

# Install tensorboard
RUN pip install --no-cache-dir tensorboard

# Set working directory
WORKDIR /app

# Expose port 6006
EXPOSE 6006

# Define volume for logs
VOLUME ["/logs"]

# Run TensorBoard
CMD ["tensorboard", "--logdir=/logs/fit", "--host=0.0.0.0", "--port=6006"]