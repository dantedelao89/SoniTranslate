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

# Create comprehensive TTS and parameter explorer
COPY <<EOF /app/handler.py
import runpod
import os
import sys
import tempfile
import subprocess
import traceback
import logging
import inspect

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
        
        # Import SoniTranslate modules
        try:
            from soni_translate.logging_setup import logger, configure_logging_libs
            configure_logging_libs()
            
            # Import the class
            from app_rvc import SoniTranslate
            
            # Create SoniTranslate instance  
            soni_translator = SoniTranslate()
            
            # EXPLORE TTS and Speaker related methods/attributes
            tts_methods = [method for method in dir(soni_translator) if any(word in method.lower() for word in ['tts', 'voice', 'speaker', 'neural'])]
            
            # Check for voice-related attributes
            voice_info = {}
            if hasattr(soni_translator, 'get_tts_voice_list'):
                try:
                    voice_info['tts_voice_list_method'] = "Available"
                except:
                    voice_info['tts_voice_list_method'] = "Method exists but failed to call"
            
            if hasattr(soni_translator, 'tts_voices'):
                try:
                    voice_info['tts_voices_attr'] = str(type(soni_translator.tts_voices))
                except:
                    voice_info['tts_voices_attr'] = "Attribute exists but failed to access"
            
            # INSPECT the multilingual_media_conversion method signature
            method = getattr(soni_translator, 'multilingual_media_conversion')
            signature = inspect.signature(method)
            
            # Get parameter details
            params_info = {}
            for param_name, param in signature.parameters.items():
                params_info[param_name] = {
                    "default": str(param.default) if param.default != inspect.Parameter.empty else "No default",
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else "No annotation",
                    "kind": str(param.kind)
                }
            
            # Look for speaker/voice parameters specifically
            speaker_params = {k: v for k, v in params_info.items() if any(word in k.lower() for word in ['speaker', 'voice', 'tts', 'neural'])}
            
            return {
                "status": "debug_comprehensive",
                "message": "Comprehensive exploration of TTS and method parameters",
                "method_signature": str(signature),
                "all_parameters": params_info,
                "speaker_voice_parameters": speaker_params,
                "tts_related_methods": tts_methods,
                "voice_info": voice_info,
                "target_voice": "es-MX-JorgeNeural-Male",
                "input_received": {
                    "video_input": video_input,
                    "target_language": target_language,
                    "hf_token_provided": bool(hf_token)
                }
            }
            
        except Exception as processing_error:
            # If processing fails, return detailed error
            return {
                "status": "error",
                "message": f"Processing failed: {str(processing_error)}",
                "error_details": traceback.format_exc(),
                "debug_info": {
                    "video_input": video_input,
                    "target_language": target_language,
                    "hf_token_provided": bool(hf_token)
                }
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

# Additional force measures - ensure no cache conflicts
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Default command
CMD ["python", "/app/handler.py"]
