# include <cuda_runtime.h>
# include <stdio.h>


__global__ void vecAddKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void vecAdd(float *A, float *B, float *C, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char **argv) {
    int n = 1024;
    float *A = (float *)malloc(n * sizeof(float));
    float *B = (float *)malloc(n * sizeof(float));
    float *C = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    vecAdd(A, B, C, n);

    for (int i = 0; i < 1000; i++) {
        printf("A[%d] = %f, B[%d] = %f, C[%d] = %f\n", i, A[i], i, B[i], i, C[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}