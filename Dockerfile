FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /content

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    rembg[gpu] \
    trimesh \
    pyglet \
    onnxruntime-gpu \
    git+https://github.com/Stability-AI/Stable-Fast-3D.git

RUN python3 -c "from rembg import new_session; new_session('u2net')"
RUN python3 -c "from tsr.system import TSR; TSR.from_pretrained('stabilityai/stable-fast-3d', config_name='config.yaml', weight_name='model.ckpt')"

COPY handler.py /content/handler.py

CMD [ "python", "-u", "/content/handler.py" ]
