# 1. Use the official lightweight Python 3.10 slim image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary Linux build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application files to the container
COPY . .

# 6. Expose Hugging Face default port
EXPOSE 7860

# 7. Start the application using Gunicorn WSGI server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]