import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    Optimized with a custom CUDA kernel for the entire forward pass.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

        # Define the CUDA kernel
        self.kernel = """
        #include <cuda_fp16.h>
        __global__ void fused_forward(const float* x, const float* weight, const float* bias, float subtract_value, float multiply_value, float* output, int batch_size, int in_features, int out_features) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < batch_size && j < out_features) {
                float sum = 0;
                for (int k = 0; k < in_features; k++) {
                    sum += x[i * in_features + k] * weight[j * in_features + k];
                }
                sum += bias[j];
                sum = (sum - subtract_value) * multiply_value;
                output[i * out_features + j] = sum > 0 ? sum : 0; 
            }
        }
        """
        self.module = torch.cuda.module_from_source(self.kernel)
        self.fused_forward = self.module.get_function("fused_forward")

    def forward(self, x):
        x = x.cuda()
        weight = self.linear.weight.cuda()
        bias = self.linear.bias.cuda()
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = weight.shape[0]

        output = torch.empty(batch_size, out_features, dtype=torch.float32, device='cuda')
        
        # Launch the kernel
        self.fused_forward(
            block=(32, 32, 1), 
            grid=(
                (batch_size + 31) // 32,
                (out_features + 31) // 32,
                1
            ),
            args=[
                x.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                self.subtract_value,
                self.multiply_value,
                output.data_ptr(),
                batch_size,
                in_features,
                out_features,
            ]
        )

        return output.cpu() 

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]