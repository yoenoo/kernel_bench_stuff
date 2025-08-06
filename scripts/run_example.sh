set -xe

# python3 scripts/generate_and_eval_single_sample.py \
#   dataset_src="huggingface" \
#   level=1 \
#   problem_id=2 \
#   max_tokens=4096 \
#   temperature=1.0 \
#   top_p=0.95 \
#   verbose=True \
#   server_type=hf \
#   model_name="cognition-ai/Kevin-32B"


# full eval
python3 scripts/generate_samples.py \
  run_name=test_hf_level_1 \
  dataset_src=huggingface \
  level=1 \
  temperature=1.0 \
  num_workers=1 \
  server_type=hf \
  model_name="cognition-ai/Kevin-32B"

## TODO: understand how this works (seems to go beyond the 10 samples)
python3 scripts/eval_from_generations.py \
  run_name=test_hf_level_1 \
  dataset_src=huggingface \
  level=1 \
  num_gpu_devices=1 \
  timeout=300




# python3 scripts/generate_and_eval_single_sample.py \
#   dataset_src="huggingface" \
#   level=1 \
#   problem_id=1 \
#   verbose=True \
#   server_type="openai" \
#   model_name="gpt-4o-2024-08-06"


  # server_type=openrouter \
  # model_name="qwen/qwen3-coder"
  # model_name="qwen/qwen3-32b"
  # model_name="qwen/qwen3-30b-a3b-instruct-2507" ## doesn't work
  # model_name="qwen/qwq-32b" ## no CUDA write? outputs gibberish