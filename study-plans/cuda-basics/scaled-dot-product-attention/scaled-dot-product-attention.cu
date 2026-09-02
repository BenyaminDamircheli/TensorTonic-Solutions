#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += Q[i * D + d] * K[j * D + d];
        }
        scores[i * N + j] = acc / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    float* r = scores + row * N; // Ptr to where we start in the scores matrix

    __shared__ float buf[256];
    __shared__ float s_sum;
    __shared__ float s_max;

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) local_max = fmax(r[j], local_max);
    buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] = fmax(buf[tid], buf[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) s_max = buf[0];
    __syncthreads();
    float m = s_max;

    float local_sum = 0.0f;

    for (int j = tid; j < N ; j += blockDim.x) {
        float e = expf(r[j] - m);
        r[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum; // reuse the buf since it's already alloc.
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) s_sum = buf[0];
    __syncthreads();
    float inv = 1.0f / s_sum;

    for (int j = tid; j < N; j += blockDim.x) r[j] *= inv;
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && d < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += attn[i * N + j] * V[j * D + d];
        }
        output[i * D + d] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}