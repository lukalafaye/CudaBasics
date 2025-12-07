#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <random>

void init_vector(float* v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX;
    }
}

void vector_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void subtract_and_sum(float *a, float* b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] - b[i];
    }
    printf("Sum: %f\n", sum);
}

// cuda kernel
__global__ void gpuAdd(const float* a, const float* b, float* c, int n) {
    int blockNumInGrid = blockIdx.z * (gridDim.y * gridDim.x) 
                       + blockIdx.y * gridDim.x 
                       + blockIdx.x; 

    int threadsPerBlock = blockDim.z * blockDim.y * blockDim.x;

    int threadNumInBlock = threadIdx.z * (blockDim.x * blockDim.y) 
                         + threadIdx.y * blockDim.x 
                         + threadIdx.x;

    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    if (globalThreadNum < n) {
        c[globalThreadNum] = a[globalThreadNum] + b[globalThreadNum];
    }
    // printf("blockNumInGrid: %d, threadsPerBlock: %d, threadNumInBlock: %d, globalThreadNum: %d \n", blockNumInGrid, threadsPerBlock, threadNumInBlock, globalThreadNum);
}

int main(int argc, char **argv) {
    int n = 2048;

    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];
    float cpu_sum[n];

    init_vector(a, n);
    init_vector(b, n);

    vector_add_cpu(a, b, cpu_sum, n);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int numThreadsPerBlock = 256;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    gpuAdd<<<numBlocks, numThreadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    subtract_and_sum(cpu_sum, c, n);

    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
