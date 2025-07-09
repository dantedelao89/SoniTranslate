# Base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_base.txt requirements_extra.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir pip==23.1.2
RUN pip install --no-cache-dir -r requirements_base.txt
RUN pip install --no-cache-dir -r requirements_extra.txt
RUN pip install --no-cache-dir onnxruntime-gpu
RUN pip install --no-cache-dir runpod

# Copy the rest of the application
COPY . .

# Create handler for RunPod serverless
RUN echo 'import runpod\n\
import os\n\
import sys\n\
\n\
# Add the current directory to Python path\n\
sys.path.append("/app")\n\
\n\
def handler(event):\n\
    """Handle RunPod serverless requests"""\n\
    try:\n\
        # Get input parameters\n\
        input_data = event.get("input", {})\n\
        \n\
        # For now, return a simple response\n\
        # TODO: Integrate with SoniTranslate main functions\n\
        return {\n\
            "status": "success",\n\
            "message": "SoniTranslate endpoint is ready",\n\
            "input_received": input_data\n\
        }\n\
    except Exception as e:\n\
        return {\n\
            "status": "error",\n\
            "message": str(e)\n\
        }\n\
\n\
if __name__ == "__main__":\n\
    runpod.serverless.start({"handler": handler})' > /app/handler.py

# Expose port (for testing)
EXPOSE 8000

# Default command
CMD ["python", "/app/handler.py"]
