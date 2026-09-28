# CUDA 12.8 + Python 3.12 + PyTorch 2.11, matching LeRobot's cu128 wheels.
# Host needs an NVIDIA driver new enough for CUDA 12.8 (driver >= 570.86).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# A plain pip install can replace this image's torch with the PyPI wheel,
# which has been CUDA 13 since PyTorch 2.11. Pin the image's own build.
RUN python -c 'import pathlib, torch, torchvision; pathlib.Path("/tmp/constraints.txt").write_text(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n")' \
    && pip install -c /tmp/constraints.txt "lerobot[diffusion,pi,training]==0.6.2" \
    && python -c 'import torch, lerobot; cuda = torch.version.cuda; assert cuda and cuda.startswith("12.8"), cuda; print(f"lerobot {lerobot.__version__}, torch {torch.__version__}, cuda {cuda}")'

COPY train.sh /usr/local/bin/train
RUN chmod +x /usr/local/bin/train

WORKDIR /workspace
ENTRYPOINT ["train"]
