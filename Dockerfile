# Base image with CUDA support (using available PyTorch image)
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

# Install Python dependencies in correct order
RUN pip install --no-cache-dir pip==23.1.2

# Install PyTorch with CUDA 11.7 (matching base image)
RUN pip install --no-cache-dir -r requirements_base.txt
RUN pip install --no-cache-dir -r requirements_extra.txt
RUN pip install --no-cache-dir onnxruntime-gpu

# Install optional but important dependencies
RUN pip install --no-cache-dir piper-tts==1.2.0
RUN pip install --no-cache-dir TTS==0.21.1 --no-deps

RUN pip install --no-cache-dir runpod

# Set environment variables including Hugging Face token
ENV YOUR_HF_TOKEN=""
ENV PYTHONPATH="/app"
ENV CUDA_VISIBLE_DEVICES="0"

# Copy the rest of the application
COPY . .

# Create improved handler for RunPod serverless
RUN echo 'import runpod\n\
import os\n\
import sys\n\
import tempfile\n\
import subprocess\n\
\n\
# Add the current directory to Python path\n\
sys.path.append("/app")\n\
\n\
def handler(event):\n\
    """Handle RunPod serverless requests for SoniTranslate"""\n\
    try:\n\
        # Get input parameters\n\
        input_data = event.get("input", {})\n\
        \n\
        # Required parameters\n\
        video_input = input_data.get("video_url") or input_data.get("video_file")\n\
        target_language = input_data.get("target_language", "en")\n\
        \n\
        if not video_input:\n\
            return {\n\
                "status": "error",\n\
                "message": "video_url or video_file is required"\n\
            }\n\
        \n\
        # Set required environment variables\n\
        hf_token = input_data.get("hf_token") or os.environ.get("YOUR_HF_TOKEN")\n\
        if hf_token:\n\
            os.environ["YOUR_HF_TOKEN"] = hf_token\n\
        \n\
        # For now, return success with input validation\n\
        # TODO: Integrate with actual SoniTranslate processing\n\
        return {\n\
            "status": "success",\n\
            "message": "SoniTranslate endpoint is ready for processing",\n\
            "input_received": {\n\
                "video_input": video_input,\n\
                "target_language": target_language,\n\
                "hf_token_provided": bool(hf_token)\n\
            },\n\
            "next_steps": "Integration with SoniTranslate core functions needed"\n\
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
