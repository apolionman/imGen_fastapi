FROM dt-base:latest

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_DTYPE=float32
ENV OPEN_CLIP_FORCE_FLOAT32=1
ENV BLIP_NUM_BEAMS=16
ENV BLIP_OFFLOAD=True
# ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app

# Install system dependencies including Rust build essentials
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    ccache \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    wget \
    pkg-config \
    libssl-dev \
    build-essential \
    libffi-dev \
    python3-dev \
    libclang-dev \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]