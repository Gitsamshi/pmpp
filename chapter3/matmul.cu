#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void matmulKernel(float *A, float *B, float *C, 
                              int height_A, int width_B, int shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height_A && col < width_B) {
        float sumValue = 0.0f;
        for (int k = 0; k < shared_dim; k++) {
            // A: row x shared_dim, B: shared_dim x col
            sumValue += A[row * shared_dim + k] * B[k * width_B + col];
        }
        C[row * width_B + col] = sumValue;
    }
}


void matmul(float *A, float *B, float *C, int height_A, int width_B, int shared_dim) {
    // A: height_A x shared_dim
    // B: shared_dim x width_B
    // C: height_A x width_B
    int size_A = height_A * shared_dim * sizeof(float);
    int size_B = shared_dim * width_B * sizeof(float);
    int size_C = height_A * width_B * sizeof(float);
    
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);
    cudaMemcpy(A_d, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_B, cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    dim3 gridDim((width_B + blockDim.x - 1) / blockDim.x, 
                 (height_A + blockDim.y - 1) / blockDim.y);
    matmulKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, height_A, width_B, shared_dim);

    cudaMemcpy(C, C_d, size_C, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void matrixMulCPU(float *A, float *B, float *C, int height_A, int width_B, int shared_dim) {
    // A: height_A x shared_dim, B: shared_dim x width_B, C: height_A x width_B
    for (int i = 0; i < height_A; i++) {
        for (int j = 0; j < width_B; j++) {
            float sumValue = 0.0f;
            for (int k = 0; k < shared_dim; k++) {
                sumValue += A[i * shared_dim + k] * B[k * width_B + j];
            }
            C[i * width_B + j] = sumValue;
        }
    }
}

int main(int argc, char **argv) {
    // For square matrices: A and B are both width x width, C is width x width
    // Can also support non-square: e.g., A is height_A x shared_dim, B is shared_dim x width_B
    int height_A = 512;  // height of A and C
    int width_B = 512;   // width of B and C
    int shared_dim = 1024; // width of A = height of B
    
    // A: height_A x shared_dim
    float *A = (float *)malloc(height_A * shared_dim * sizeof(float));
    // B: shared_dim x width_B
    float *B = (float *)malloc(shared_dim * width_B * sizeof(float));
    // C: height_A x width_B
    float *C = (float *)malloc(height_A * width_B * sizeof(float));
    
    // Initialize matrices (example for square matrices)
    for (int i = 0; i < height_A; i++) {
        for (int j = 0; j < shared_dim; j++) {
            A[i * shared_dim + j] = i + j;
        }
    }
    for (int i = 0; i < shared_dim; i++) {
        for (int j = 0; j < width_B; j++) {
            B[i * width_B + j] = i - j;
        }
    }
    
    printf("Multiply the matrices (A: %dx%d, B: %dx%d, C: %dx%d)\n", 
           height_A, shared_dim, shared_dim, width_B, height_A, width_B);
    matmul(A, B, C, height_A, width_B, shared_dim);
    printf("Done!\n");
    printf("Sample of the result:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("C[%d, %d] = %f\n", i, j, C[i * width_B + j]);
        }
    }
    printf("Compare the result with the CPU version\n");
    float *C_cpu = (float *)malloc(height_A * width_B * sizeof(float));
    matrixMulCPU(A, B, C_cpu, height_A, width_B, shared_dim);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("C[%d, %d] = %f, C_cpu[%d, %d] = %f\n", i, j, C[i * width_B + j], i, j, C_cpu[i * width_B + j]);
        }
    }
    free(C_cpu);
    free(A);
    free(B);
    free(C);
    return 0;
}