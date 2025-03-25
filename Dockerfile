# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the current directory contents into the container
COPY . /app

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Expose ports (5000 for Flask, 7860 for Gradio)
EXPOSE 5000
EXPOSE 7860

# Step 7: Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Step 8: Run both Flask (via Gunicorn) and Gradio (via a background process)
CMD gunicorn --bind 0.0.0.0:5000 --workers 4 app:app & python gradio_app.py
