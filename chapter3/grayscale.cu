# include <cuda_runtime.h>
# include <stdio.h>

#define CHANNELS 3

__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row * width + col;

        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];
        unsigned char b = input[rgbOffset + 2];

        unsigned char gray = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);

        output[grayOffset] = gray;
    }
}


void colorToGrayscale(unsigned char *input, unsigned char *output, int width, int height) {
    int colorsize = width * height * CHANNELS * sizeof(unsigned char);
    int graysize = width * height * sizeof(unsigned char);

    unsigned char *input_d, *output_d;
    cudaMalloc((void **)&input_d, colorsize);
    cudaMalloc((void **)&output_d, graysize);
    cudaMemcpy(input_d, input, colorsize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);



    grayscaleKernel<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    cudaMemcpy(output, output_d, graysize, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}


int main(int argc, char **argv) {
    int width = 1024;
    int height = 1024;
    unsigned char*input = (unsigned char*)malloc(width * height * CHANNELS * sizeof(unsigned char));
    unsigned char*output = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * CHANNELS;
            input[index] = rand() % 256;
            input[index + 1] = rand() % 256;
            input[index + 2] = rand() % 256;
        }
    }

    printf ("convert color to grayscale");

    colorToGrayscale(input, output, width, height);
    printf("\n Sample of grayscale image:\n");

    for (int i = 0; i < 10; i++) {
        int index = (i * width + i) * CHANNELS;
        int grayIndex = (i * width + i);
        printf("input[%d] = %d, input[%d] = %d, input[%d] = %d, output[%d] = %d\n", index, input[index], index + 1, input[index + 1], index + 2, input[index + 2], grayIndex, output[grayIndex]);
    }
    free(input);
    free(output);

    printf("Done!\n");
    return 0;
}
