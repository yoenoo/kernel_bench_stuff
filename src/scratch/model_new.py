import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_clamp_mul_clamp(float* input, float* bias, float* output, int size, float scaling_factor, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + bias[idx];
        val = fmaxf(min_val, fminf(max_val, val));
        val = val * scaling_factor;
        val = fmaxf(min_val, fminf(max_val, val));
        output[idx] = val / scaling_factor;
    }
}

torch::Tensor fused_clamp_mul_clamp_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    int size = input.numel();
    torch::Tensor output = torch::zeros_like(input);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    fused_clamp_mul_clamp<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        scaling_factor,
        0.0f,
        1.0f
    );

    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_clamp_mul_clamp_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_kernel = load_inline(
    name='fused_clamp_mul_clamp',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_clamp_mul_clamp_cuda'],
    verbose=True,
    extra_cflags=['-I/usr/local/cuda/include'],
    extra_ldflags=['-L/usr/local/cuda/lib64']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_kernel.fused_clamp_mul_clamp_cuda(x, self.bias.expand_as(x), self.scaling_factor)
        return x

# # Example usage
# batch_size = 128
# in_channels = 3
# out_channels = 16
# height, width = 32, 32
# kernel_size = 3
# stride = 2
# padding = 1
# output_padding = 1
# bias_shape = (out_channels, 1, 1)
# scaling_factor = 2.0

# def get_inputs():
#     return [torch.randn(batch_size, in_channels, height, width)]

# def get_init_inputs():
#     return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

# # Instantiate the model
# model = ModelNew(*get_init_inputs())
# input_tensor = get_inputs()[0]
# output_tensor = model(input_tensor)
# print(output_tensor.shape)