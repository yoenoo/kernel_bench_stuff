import subprocess
import os, sys

from utils import read_file, extract_first_code
'''
Construct Prompts
As basic as we can be, not to steer the LLM too much
'''

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)
    
def get_arch_definition(arch_src):
    '''
    Construct torch definition from original torch nn.Module definition 
    '''
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
def prompt_generate_custom_cuda(arc_src: str,
                                 example_arch_src: str,
                                 example_new_arch_src: str) -> str:
    prompt = f"""
    I want to write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.
    You have complete freedom to choose the operators you want to replace. You may replace multiple operators with custom implementations, consider operator fusion opportunities (such as combining matmul and relu into a single kernel), or algorithmic changes (such as efficient online softmax). You are only limited by your imagination.\n
    Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example arch looks like this: ```{example_arch_src}```. \n
    The example new arch with custom CUDA kernels looks like this: ```{example_new_arch_src}```. \n
    You are given the following architecture: ```{arc_src}```. Optimize it with custom CUDA operators! If the old architecture is called <Model>, name this one <Model>New. Output the new code in codeblocks.\n
    """
    return prompt

def prompt_generate_custom_cuda_from_file(arch_path):
    arch = get_arch_definition_from_file(arch_path)
    # These are strictly defined for now
    
    example_arch_path = os.path.join(KERNEL_BENCH_PATH, "tests/prompts/model_ex.py")
    example_new_arch_path = os.path.join(KERNEL_BENCH_PATH, "tests/prompts/model_new_ex.py")
    
    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(f"Example new architecture file not found: {example_new_arch_path}")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)
    

def check_prompt_generate_custom_cuda(arch_path):
    generated_prompt = prompt_generate_custom_cuda_from_file(arch_path)

    os.makedirs("./scratch", exist_ok=True)
    with open("./scratch/prompt.txt", "w") as f:
        f.write(generated_prompt)

    return generated_prompt

def run(arch_path):
    # read in an architecture file, copy it to ./scratch/model.py
    arch = read_file(arch_path)
    # Ensure the ./scratch directory exists
    os.makedirs("./scratch", exist_ok=True)
    
    # Write the architecture to ./scratch/model.py
    with open("./scratch/model.py", "w") as f:
        f.write(arch)

    # # generate test harness, save in ./scratch/test.py
    # test_harness = query_server(prompt_generate_test_harness_from_file(arch_path), server_type=SERVER_TYPE)
    # test_harness = extract_first_code(test_harness, "python")
    # with open("./scratch/test.py", "w") as f:
    #     f.write(test_harness)
    # generate custom CUDA, save in ./scratch/model_new.py
    custom_cuda_prompt = prompt_generate_custom_cuda_from_file(arch_path)
    custom_cuda = run_llm(custom_cuda_prompt)

    # import pdb; pdb.set_trace()
    custom_cuda = extract_first_code(custom_cuda, "python")
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    print("[Verification] Torch moduel with Custom CUDA code **GENERATED** successfully")

    # check if the generated code compiles
    try:
        exec(custom_cuda)
        print("[Verification] Custom CUDA code **COMPILES** successfully")
    except Exception as e:
        raise RuntimeError(f"Error compiling generated custom cuda code: {e}")

    with open("./scratch/model_new.py", "w") as f:
        f.write(custom_cuda)

    # check generated code is correct / functionally equivalent.
    # run test harness, save output in log.txt
    with open('./scratch/log.txt', 'w') as log_file:
        process = subprocess.run(
            ['python', './scratch/test.py'], 
            stdout=log_file,  # Redirect stdout to log.txt
            stderr=subprocess.STDOUT  # Redirect stderr to log.txt as well
        )
        if process.returncode == 0:
            print("[Verification] Custom CUDA kernel is **Correct**, matches reference")
            return "PASS"
        else:
            print("[Verification] Custom CUDA kernel **FAIL** to match reference in terms of correctness")
            return "FAIL"
