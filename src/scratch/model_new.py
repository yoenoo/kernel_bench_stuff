import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.dtype(), A.device());
    const int block_size = 16;
    const int num_blocks_x = (M + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;
    matmul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name='matmul',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for sum + max + avg pool + logsumexp fusion
fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fusion_kernel(const float* x, float* out, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum = 0.0f;
        float max_val = x[i];
        for (int j = 0; j < 5; j++) {
            sum += x[i + j * batch_size];
            if (x[i + j * batch_size] > max_val) {
                max_val = x[i + j * batch_size];
            }
        }
        out[i] = max_val;
        out[i] = out[i] / 5.0f; // Average pooling
        out[i] = expf(out[i]);
        out[i] = logf(out[i]);
    }
}

torch::Tensor fusion_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    fusion_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size);
    return out;
}
"""

fusion_cpp_source = "torch::Tensor fusion_cuda(torch::Tensor x);"

# Compile the inline CUDA code for fusion kernel
fusion = load_inline(
    name='fusion',
    cpp_sources=fusion_cpp_source,
    cuda_sources=fusion_source,
    functions=['fusion_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul = matmul
        self.fusion = fusion

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.matmul.matmul_cuda(x, self.linear.weight.T)  # (batch_size, out_features)
        x = self.fusion.fusion_cuda(x)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x