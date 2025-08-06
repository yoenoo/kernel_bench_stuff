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
# default config: Starting Batch Generation with config: EvalConfig({'dataset_src': 'huggingface', 'dataset_name': 'ScalingIntelligence/KernelBench', 'level': 1, 'subset': [None, None], 'run_name': 'test_hf_level_1', 'num_workers': 50, 'api_query_interval': 0.0, 'server_type': 'hf', 'model_name': 'cognition-ai/Kevin-32B', 'max_tokens': 4096, 'temperature': 0.0, 'runs_dir': '/root/kernel_bench_stuff/runs', 'verbose': False, 'store_type': 'local', 'log_prompt': False})
python3 scripts/generate_samples.py \
  run_name=test_hf_level_1 \
  dataset_src=huggingface \
  level=1 \
  num_workers=50 \
  server_type=hf \
  model_name="cognition-ai/Kevin-32B"

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