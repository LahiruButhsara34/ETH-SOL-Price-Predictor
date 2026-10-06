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

# 6. Expose application port
#EXPOSE 5000

# 7. Start the application 
#CMD ["python", "main.py"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app"]