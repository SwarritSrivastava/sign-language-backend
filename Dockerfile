# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy project files
COPY . /app/

# Create and activate a virtual environment
RUN python -m venv /opt/venv

# Install dependencies inside the venv
RUN /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

# Set environment variables to use venv
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port your app runs on (if needed)
EXPOSE 5000

# Command to start your Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
