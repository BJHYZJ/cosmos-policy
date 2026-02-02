# 1. (Optional) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2. (Optional) install cuda-12.1
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
rm -rf cuda_12.8.0_570.86.10_linux.run
## set ~/.bashrc to overwrite cuda-12.8


# 3. install environment
UV_PYTHON=python3.10 ./bin/uv_lock.sh .
uv pip install -e .
## libero
uv sync --extra cu128 --group libero --python 3.10


## robocasa
uv sync --extra cu128 --group robocasa  --python 3.10
uv pip install -e .
### Then, clone the RoboCasa repo, install the package, and download assets
cd ../  # we suggest install robocasa in the outside of the repo
git clone https://github.com/robocasa/robocasa.git  # test in `756598a`
cd robocasa
### remove numpy version assert in `robocasa/robocasa/__init__.py`
sed -i '311,318s/^/# /' robocasa/__init__.py
source ../cosmos-policy/.venv/bin/activate

uv pip install -e .
uv pip install pre-commit; pre-commit install
uv run --extra cu128 --group robocasa --python 3.10 robocasa/scripts/download_kitchen_assets.py

# (Optional) download the dataset
cd /path/to/your/dataset_path
hf download nvidia/RoboCasa-Cosmos-Policy --repo-type dataset --local-dir RoboCasa-Cosmos-Policy




## libero plus
uv sync --extra cu128 --group libero --python 3.10
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus
uv pip install -e .
uv pip install -r extra_requirements.txt