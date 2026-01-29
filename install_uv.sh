# 1. (Optional) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2. (Optional) install cuda-12.1
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
rm -rf cuda_12.8.0_570.86.10_linux.run
## set ~/.bashrc to overwrite cuda-12.8

# 3. create python 3.10 environment
# create uv.lock just for you environment
UV_PYTHON=python3.10 ./bin/uv_lock.sh .
uv sync --extra cu128 --group libero --python 3.10


# If you meet package not found error, try:
cd ./.venv/lib/python3.10/site-packages
ln -s transformer_engine-2.2+cu128.torch27.dist-info transformer_engine_torch-2.2+cu128.torch27.dist-info