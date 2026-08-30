# Single image for all three services; the command selects the role.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY common ./common
COPY services ./services

EXPOSE 8000

# Default role is the router; compose/k8s override --app-dir per service.
CMD ["uvicorn", "main:app", "--app-dir", "services/router", \
     "--host", "0.0.0.0", "--port", "8000"]
