import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# load model and the modified model
from model import Model
from model import get_inputs
from model import get_init_inputs
from model_new import ModelNew

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def check_correctness():
    # run the model and check correctness
    with torch.no_grad():
        set_seed(42)
        inputs = get_inputs()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        set_seed(42)
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]

        set_seed(42)
        model = Model(*init_inputs).cuda()

        set_seed(42)
        model_new = ModelNew(*init_inputs).cuda()

        assert(output.shape == output_new.shape)
        assert(torch.allclose(output, output_new, atol=1e-02))

    return "PASS"

def run(random_seed=42):

    # run both models and check correctness
    check_correctness()

    return "PASS"

if __name__ == "__main__":
    print(run())