# include <cuda_runtime.h>
# include <stdio.h>
# include <stdlib.h>

#define KERNEL_SIZE 2

__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        int pixValue = 0;
        int pixelCount = 0;
        for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
            for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
                int neighborX = col + i;
                int neighborY = row + j;
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    int neighborIndex = neighborY * width + neighborX;
                    pixValue += input[neighborIndex];
                    pixelCount++;
                }
            }
        }
        output[index] = pixValue / pixelCount;
    }
}

void blur(unsigned char *input, unsigned char *output, int width, int height) {
    int size = width * height * sizeof(unsigned char);
    unsigned char *input_d, *output_d;
    cudaMalloc((void **)&input_d, size);
    cudaMalloc((void **)&output_d, size);
    cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    blurKernel<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(int argc, char **argv) {
    int width = 1024;
    int height = 1024;
    unsigned char *input = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *output = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i * width + j] = rand() % 256;
        }
    }
    printf("Blur the image\n");
    blur(input, output, width, height);
    printf("Done!\n");
    printf("Sample of blurred image:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("input[%d, %d] = %d, output[%d, %d] = %d\n", i, j, input[i * width + j], i, j, output[i * width + j]);
        }
    }
    free(input);
    free(output);
    return 0;
}