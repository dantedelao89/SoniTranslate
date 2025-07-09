# Base image with CUDA support (using available PyTorch image)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

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

RUN pip install --no-cache-dir -r requirements_base.txt
RUN pip install --no-cache-dir -r requirements_extra.txt
RUN pip install --no-cache-dir onnxruntime-gpu

# Install optional but important dependencies
RUN pip install --no-cache-dir piper-tts==1.2.0
RUN pip install --no-cache-dir TTS==0.21.1 --no-deps

RUN pip install --no-cache-dir runpod

# Set environment variables (token will be added via RunPod Environment Variables)
ENV PYTHONPATH="/app"
ENV CUDA_VISIBLE_DEVICES="0"

# Copy the rest of the application
COPY . .

# Create improved handler for RunPod serverless
COPY <<EOF /app/handler.py
import runpod
import os
import sys
import tempfile
import subprocess
import traceback
import logging

# Add the current directory to Python path
sys.path.append("/app")

def handler(event):
    """Handle RunPod serverless requests for SoniTranslate"""
    try:
        # Get input parameters
        input_data = event.get("input", {})
        
        # Required parameters
        video_input = input_data.get("video_url") or input_data.get("video_file")
        target_language = input_data.get("target_language", "en")
        
        if not video_input:
            return {
                "status": "error",
                "message": "video_url or video_file is required"
            }
        
        # Set HF token from environment or input
        hf_token = input_data.get("hf_token") or os.environ.get("YOUR_HF_TOKEN")
        if hf_token:
            os.environ["YOUR_HF_TOKEN"] = hf_token
        
        # Optional parameters with defaults
        source_language = input_data.get("source_language", "auto")
        speaker_voice = input_data.get("speaker_voice", "auto")
        output_type = input_data.get("output_type", "mp4")
        
        # Import SoniTranslate modules
        try:
            from soni_translate.logging_setup import logger, configure_logging_libs
            configure_logging_libs()
            
            # Import main processing class
            from app_rvc import SoniTranslateClass
            
            # Create SoniTranslate instance
            soni_translator = SoniTranslateClass()
            
            # Process the video
            result = soni_translator.media_file(
                media_file=video_input,
                source_language=source_language,
                target_language=target_language,
                speaker_voice=speaker_voice,
                output_type=f"video ({output_type})",
                whisper_model_default="large-v3",
                compute_type="int8",
                batch_size=16,
                is_gui=False,
                progress=None
            )
            
            return {
                "status": "success",
                "message": "Video translation completed",
                "result": {
                    "translated_video": result,
                    "source_language": source_language,
                    "target_language": target_language,
                    "output_type": output_type
                }
            }
            
        except ImportError as e:
            # If import fails, try alternative approach
            return {
                "status": "error", 
                "message": f"Failed to import SoniTranslate modules: {str(e)}",
                "debug": "Modules may not be properly installed or configured"
            }
            
    except Exception as e:
        # Capture full error details
        error_details = traceback.format_exc()
        return {
            "status": "error",
            "message": str(e),
            "error_details": error_details,
            "input_received": input_data
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
EOF

# Expose port (for testing)
EXPOSE 8000

# Default command
CMD ["python", "/app/handler.py"]
