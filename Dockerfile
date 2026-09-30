FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /content

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Stability-AI/Stable-Fast-3D.git /content/sf3d

WORKDIR /content/sf3d

RUN pip install --no-cache-dir \
    runpod \
    rembg[gpu] \
    trimesh \
    pyglet \
    onnxruntime-gpu \
    -r requirements.txt

WORKDIR /content

COPY handler.py /content/handler.py

CMD [ "python", "-u", "/content/handler.py" ]
