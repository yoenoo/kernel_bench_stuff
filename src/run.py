import subprocess
import os, sys
from utils import query_server, read_file, extract_first_code
from prompt_constructor import prompt_generate_custom_cuda_from_file, prompt_generate_custom_cuda_from_file_save

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

SERVER_TYPE = "gemini"

server_args = {
    "deepseek": {
        "temperature": 1.6,
        "max_tokens": 4096
    },
    "gemini": {}, # need to experiment with temperature,
    "together": { # this is Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096
    },
    "sglang": { # this is Llama 3
        "temperature": 0.7,
    }
}

def run_llm(prompt):
    '''
    Call use common API query function with monkeys
    '''
    return query_server(prompt, server_type=SERVER_TYPE
                        , **server_args[SERVER_TYPE])

def run(arch_path, save_prompt=False, prompt_example_ind=0):
    # read in an architecture file, copy it to REPO_TOP_PATH/src/scratch/model.py
    arch = read_file(arch_path)
    # Ensure the scratch directory exists
    os.makedirs(os.path.join(REPO_TOP_PATH, "src/scratch"), exist_ok=True)
    # Write the architecture to REPO_TOP_PATH/src/scratch/model.py
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model.py"), "w") as f:
        f.write(arch)

    # generate custom CUDA, save in scratch/model_new.py
    fn_get_prompt = prompt_generate_custom_cuda_from_file_save if save_prompt else prompt_generate_custom_cuda_from_file
    custom_cuda_prompt = fn_get_prompt(arch_path, prompt_example_ind)
    custom_cuda = run_llm(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, "python")

    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    print("[Verification] Torch module with Custom CUDA code **GENERATED** successfully")

    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model_new.py"), "w") as f:
        f.write(custom_cuda)

    # check if the generated code compiles
    try:
        exec(custom_cuda, globals())
        print("[Verification] Custom CUDA code **COMPILES** successfully")
    except Exception as e:
        raise RuntimeError(f"Error compiling generated custom cuda code: {e}")

    # check generated code is correct / functionally equivalent.
    # run test harness, save output in log.txt
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/log.txt"), 'w') as log_file:
        process = subprocess.run(
            ['python', os.path.join(REPO_TOP_PATH, "src/scratch/test.py")], 
            stdout=log_file,  # Redirect stdout to log.txt
            stderr=subprocess.STDOUT  # Redirect stderr to log.txt as well
        )
        if process.returncode == 0:
            print("[Verification] Custom CUDA kernel is **Correct**, matches reference")
            return "PASS"
        else:
            print("[Verification] Custom CUDA kernel **FAIL** to match reference in terms of correctness")
            return "FAIL"

# if __name__ == "__main__":
    # run(os.path.join(KERNEL_BENCH_PATH, "level1/17_Matmul_with_transposed_B.py"))
    # run(os.path.join(KERNEL_BENCH_PATH, "level2/9_Matmul_Subtract_Multiply_ReLU.py"))
    # run(os.path.join(KERNEL_BENCH_PATH, "level3/45_MiniGPTBlock.py"))