# DINO by Facebook/Meta and Sorbonne

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM nvcr.io/nvidia/pytorch:21.05-py3
# ^ fails on nvidia-driver 460, but 20.12 works
# FROM nvcr.io/nvidia/pytorch:20.12-py3
# ^ got an error with 'interpolate' in tv.transforms.RandomRotate???
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install linux packages
RUN apt update && apt install -y libgl1

# Install python dependencies
# COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache timm
# RUN pip install --no-cache -U torch torchvision numpy
# RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /project

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/project

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3
