# 1. (Optional) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2. (Optional) install cuda-12.1
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
rm -rf cuda_12.8.0_570.86.10_linux.run
## set ~/.bashrc to overwrite cuda-12.8

# 3. create python 3.10 environment
# uv venv --python 3.10
uv venv --python 3.10

# 4. activate environment and install environments
source .venv/bin/activate
uv sync --active --extra cu128 --group libero

# deactivate environment
deactivate