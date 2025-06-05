FROM python:3.10-slim

RUN pip install tensorboard

WORKDIR /app
VOLUME ["/logs"]

CMD ["tensorboard", "--logdir=/logs", "--host=0.0.0.0", "--port=6006"]
