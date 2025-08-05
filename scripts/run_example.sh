set -xe

# python3 -m sglang.launch_server \
#   --model-path "cognition-ai/Kevin-32B" \  # example HF/local path
#   --host "localhost" \
#   --port 30000 \

# python3 scripts/generate_and_eval_single_sample.py \
#   dataset_src="huggingface" \
#   level=1 \
#   problem_id=6 \
#   verbose=True \
#   server_type=hf \
#   model_name="cognition-ai/Kevin-32B"
#   # model_name="qwen/qwen3-30b-a3b-instruct-2507" ## doesn't work
#   # model_name="qwen/qwen3-coder"

# exit 0


python3 scripts/generate_and_eval_single_sample.py \
  dataset_src="huggingface" \
  level=1 \
  problem_id=4 \
  verbose=True \
  server_type="openai" \
  model_name="gpt-4o-2024-08-06"
  # server_type=openrouter \
  # model_name="qwen/qwen3-coder"
  # model_name="qwen/qwen3-32b"
  # model_name="qwen/qwen3-30b-a3b-instruct-2507" ## doesn't work
  # model_name="qwen/qwq-32b" ## no CUDA write? outputs gibberish