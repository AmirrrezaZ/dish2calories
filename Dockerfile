# PyTorch base image with CUDA 11.8 + cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Copy project files
COPY ./ /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpng-dev \
        libjpeg-dev \
        zlib1g-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# (Optional) show these ports in docker ps; compose still maps them
EXPOSE 8000
EXPOSE 8501

# (Optional) avoid Streamlit asking for a browser
ENV STREAMLIT_SERVER_HEADLESS=true
