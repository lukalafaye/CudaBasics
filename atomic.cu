#include <stdio.h> 
#include <cuda_runtime.h>

#define NUM_BLOCKS 1000
#define NUM_THREADS 32

__global__ void addCounter(int* counter) {
    *counter += 1;
}

__global__ void atomicCounter(int* counter) {
    atomicAdd(counter, 1);
}

int main(int argc, char** argv) {
    int counter_nonatomic = 0;
    int counter_atomic = 0;

    int* d_counter_nonatomic;
    int* d_counter_atomic;

    cudaMalloc(&d_counter_nonatomic, sizeof(int));
    cudaMalloc(&d_counter_atomic, sizeof(int));
    
    cudaMemcpy(d_counter_nonatomic, &counter_nonatomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter_atomic, &counter_atomic, sizeof(int), cudaMemcpyHostToDevice);
    
    addCounter<<<NUM_BLOCKS, NUM_THREADS>>>(d_counter_nonatomic);
    atomicCounter<<<NUM_BLOCKS, NUM_THREADS>>>(d_counter_atomic);

    cudaMemcpy(&counter_nonatomic, d_counter_nonatomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&counter_atomic, d_counter_atomic, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Non-atomic counter value: %d\n", counter_nonatomic);
    printf("Atomic counter value: %d\n", counter_atomic);
}
