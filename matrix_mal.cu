#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <random>

#define M 256 // number of rows in A and C
#define K 512 // number of columns in A and rows in B
#define N 512 // number of columns in B and C

#define BLOCK_SIZE 32

void init_matrix(float *A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i*cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

void matmul_cpu(float *A, float *B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {        // rows over A
        for (int l = 0; l < n; l++) {    // columns over B
            float sum = 0.0;
            for (int j = 0; j < k; j++) { // element wise mult
                sum += A[i*k + j] * B[j*n + l];
            }
            C[i*n + l] = sum;
        }
    }
}

void subtract_and_sum_matrix(float *A, float* B, int n, int k) {
    float sum = 0.0;
    for (int i = 0; i < n*k; i++) {
        sum += A[i] - B[i];
    }
    printf("Sum: %f\n", sum);
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        // we want to fill C[row][col]
        float sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row*k + i] * B[i*n + col];
        }
        C[row*n + col] = sum;
    }
}

int main(int argc, char **argv) {
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C_cpu = new float[M*N];
    float* C_gpu = new float[M*N];

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    matmul_cpu(A, B, C_cpu, M, K, N);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsInBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid2d((BLOCK_SIZE + N - 1) / BLOCK_SIZE, 
                (BLOCK_SIZE + M - 1) / BLOCK_SIZE);

    matmul_gpu<<<grid2d, threadsInBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    subtract_and_sum_matrix(C_cpu, C_gpu, N, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
}
