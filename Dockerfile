# GPU image (CUDA runtime, much smaller than the -devel variant).
# For CPU-only hosts use Dockerfile.cpu instead.
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# The base image already ships CUDA PyTorch, so skip the torch lines of requirements.txt.
COPY requirements.txt requirements-api.txt ./
RUN grep -vE '^(torch|torchvision|--extra-index-url)' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt -r requirements-api.txt

COPY . .

EXPOSE 8000 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

# Default: API + UI in one container (see start.sh). Compose overrides this.
CMD ["bash", "start.sh"]
