from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "cognition-ai/Kevin-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype=torch.bfloat16,
  device_map="auto"
)

# Define system prompt
system_prompt = "You are a helpful AI assistant. Provide accurate and informative responses."

# User prompt
user_prompt = "What is the capital of France?"

# Use the tokenizer's chat template for proper formatting
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# Apply the chat template - this handles the formatting automatically
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, temperature=1.2, max_new_tokens=128)

# Extract only the newly generated tokens (exclude the input prompt)
input_length = model_inputs["input_ids"].shape[1]
generated_response_ids = generated_ids[0][input_length:]

# Decode only the generated response
response = tokenizer.decode(generated_response_ids, skip_special_tokens=True)
print(response)