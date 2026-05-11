# Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app
COPY inference_server.py .
RUN pip install fastapi uvicorn nnsight transformers \
                accelerate einops --no-cache-dir

# Pre-download model weights into image
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')"

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]