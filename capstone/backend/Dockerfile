FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Copy requirements first, separately from the rest of the code, so Docker
# caches this layer — if requirements.txt hasn't changed, rebuilds skip
# reinstalling every package, which is a huge time saver during development
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# --host 0.0.0.0 is required inside Docker: the default (127.0.0.1) would
# only accept connections from inside the container itself, blocking
# anything from outside it — including other containers and your browser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
