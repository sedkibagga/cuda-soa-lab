# CUDA-enabled FastAPI Service Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && \
	apt-get install -y python3 python3-pip python3-venv python3-dev build-essential && \
	rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip3 install --upgrade pip && \
	pip3 install fastapi uvicorn[standard] numpy numba python-multipart

# Expose FastAPI port (change <student_port> to your assigned port, e.g., 8001)
EXPOSE 8001

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
