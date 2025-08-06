# run "source env_setup.sh"
set -xe

curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

uv venv --python=3.10
. .venv/bin/activate

uv pip install torch transformers accelerate datasets 
uv pip install --upgrade datasets
uv pip install python-dotenv pydra-config
uv pip install together openai anthropic google-generativeai
uv pip install ninja
uv pip install -e .

apt-get update && apt-get install -y python3.10-dev build-essential