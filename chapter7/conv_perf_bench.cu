// ============================================================================
// CUDA卷积算法完整对比 - 所有内存访问策略
// 包括：直接内存访问、L1/L2缓存、纹理缓存、共享内存等
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>
#include <assert.h>

// ============================================================================
// 配置参数
// ============================================================================
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + 2 * FILTER_RADIUS)

// 用于测试的多个问题规模
const int TEST_SIZES[] = {256, 512, 1024, 2048};
const int NUM_TEST_SIZES = 4;

// 常量内存中的卷积核
__constant__ float d_Filter[FILTER_SIZE * FILTER_SIZE];

// ============================================================================
// 策略0: CPU参考实现（用于验证）
// ============================================================================
void convolution_cpu_reference(
    float *input,
    float *output,
    float *filter,
    int width,
    int height)
{
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
                for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
                    int in_x = x + fx;
                    int in_y = y + fy;
                    
                    if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + 
                                       (fx + FILTER_RADIUS);
                        sum += input[in_y * width + in_x] * filter[filter_idx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

// ============================================================================
// 策略1: 直接全局内存访问（无任何优化）
// 特点：每个线程独立计算，没有数据重用，完全依赖DRAM带宽
// ============================================================================
__global__ void convolution2D_direct_naive(
    float *input,
    float *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= width || y >= height) return;
    
    float result = 0.0f;
    
    // 每次访问都从DRAM读取，没有任何缓存优化
    #pragma unroll
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        #pragma unroll
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            int in_x = x + fx;
            int in_y = y + fy;
            
            if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                // 直接从全局内存读取，可能触发多次DRAM访问
                float pixel = input[in_y * width + in_x];
                int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
                result += pixel * d_Filter[filter_idx];
            }
        }
    }
    
    output[y * width + x] = result;
}

// ============================================================================
// 策略2: 使用L1缓存（通过__ldg内联函数）
// 特点：显式使用只读缓存路径，减少缓存污染
// ============================================================================
__global__ void convolution2D_L1_cache(
    const float * __restrict__ input,
    float *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= width || y >= height) return;
    
    float result = 0.0f;
    
    #pragma unroll
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        #pragma unroll
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            int in_x = x + fx;
            int in_y = y + fy;
            
            if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                // 使用__ldg强制通过只读缓存路径
                float pixel = __ldg(&input[in_y * width + in_x]);
                int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
                result += pixel * d_Filter[filter_idx];
            }
        }
    }
    
    output[y * width + x] = result;
}

// ============================================================================
// 策略3: 使用L2缓存预取
// 特点：使用预取指令提高L2缓存命中率
// ============================================================================
__global__ void convolution2D_L2_prefetch(
    const float * __restrict__ input,
    float *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= width || y >= height) return;
    
    // 预取下一行数据到L2缓存
    if(y + FILTER_RADIUS < height) {
        __builtin_prefetch(&input[(y + FILTER_RADIUS) * width + x], 0, 3);
    }
    
    float result = 0.0f;
    
    #pragma unroll
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        #pragma unroll
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            int in_x = x + fx;
            int in_y = y + fy;
            
            if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                float pixel = input[in_y * width + in_x];
                int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
                result += pixel * d_Filter[filter_idx];
            }
        }
    }
    
    output[y * width + x] = result;
}

// ============================================================================
// 策略4: 纹理内存（使用纹理对象）
// 特点：专门的纹理缓存，优化2D空间局部性
// ============================================================================
__global__ void convolution2D_texture(
    cudaTextureObject_t texInput,
    float *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= width || y >= height) return;
    
    float result = 0.0f;
    
    #pragma unroll
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        #pragma unroll
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            // 纹理内存自动处理边界和缓存
            float pixel = tex2D<float>(texInput, x + fx + 0.5f, y + fy + 0.5f);
            int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
            result += pixel * d_Filter[filter_idx];
        }
    }
    
    output[y * width + x] = result;
}

// ============================================================================
// 策略5: 共享内存（完整加载包括Halo）
// 特点：最大化数据重用，但共享内存使用量大
// ============================================================================
__global__ void convolution2D_shared_full(
    float *input,
    float *output,
    int width,
    int height)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // 计算输入块的起始位置
    int in_start_x = blockIdx.x * TILE_SIZE - FILTER_RADIUS;
    int in_start_y = blockIdx.y * TILE_SIZE - FILTER_RADIUS;
    
    // 协作加载整个块（包括halo）
    for(int j = ty; j < BLOCK_SIZE; j += blockDim.y) {
        for(int i = tx; i < BLOCK_SIZE; i += blockDim.x) {
            int global_x = in_start_x + i;
            int global_y = in_start_y + j;
            
            if(global_x >= 0 && global_x < width && 
               global_y >= 0 && global_y < height) {
                tile[j][i] = input[global_y * width + global_x];
            } else {
                tile[j][i] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // 从共享内存计算
    if(out_x < width && out_y < height && tx < TILE_SIZE && ty < TILE_SIZE) {
        float result = 0.0f;
        
        #pragma unroll
        for(int fy = 0; fy < FILTER_SIZE; fy++) {
            #pragma unroll
            for(int fx = 0; fx < FILTER_SIZE; fx++) {
                result += tile[ty + fy][tx + fx] * 
                         d_Filter[fy * FILTER_SIZE + fx];
            }
        }
        
        output[out_y * width + out_x] = result;
    }
}

// ============================================================================
// 策略6: 共享内存（仅内部）+ L1/L2缓存（Halo）
// 特点：平衡共享内存使用和缓存效率
// ============================================================================
__global__ void convolution2D_shared_hybrid(
    float *input,
    float *output,
    int width,
    int height)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // 只加载内部元素到共享内存
    if(out_x < width && out_y < height) {
        tile[ty][tx] = input[out_y * width + out_x];
    } else {
        tile[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    if(out_x < width && out_y < height) {
        float result = 0.0f;
        
        #pragma unroll
        for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
            #pragma unroll
            for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
                int in_x = out_x + fx;
                int in_y = out_y + fy;
                int tile_x = tx + fx;
                int tile_y = ty + fy;
                
                float value;
                
                if(tile_x >= 0 && tile_x < TILE_SIZE && 
                   tile_y >= 0 && tile_y < TILE_SIZE) {
                    // 内部元素：从共享内存
                    value = tile[tile_y][tile_x];
                } else if(in_x >= 0 && in_x < width && 
                         in_y >= 0 && in_y < height) {
                    // Halo：从全局内存（期望缓存命中）
                    value = __ldg(&input[in_y * width + in_x]);
                } else {
                    // Ghost：边界外
                    value = 0.0f;
                }
                
                int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + 
                                (fx + FILTER_RADIUS);
                result += value * d_Filter[filter_idx];
            }
        }
        
        output[out_y * width + out_x] = result;
    }
}

// ============================================================================
// 策略7: 寄存器阻塞（Register Blocking）
// 特点：每个线程计算多个输出，最大化寄存器重用
// ============================================================================
#define REG_TILE_X 2
#define REG_TILE_Y 2

__global__ void convolution2D_register_blocking(
    float *input,
    float *output,
    int width,
    int height)
{
    int base_x = (blockIdx.x * blockDim.x + threadIdx.x) * REG_TILE_X;
    int base_y = (blockIdx.y * blockDim.y + threadIdx.y) * REG_TILE_Y;
    
    float results[REG_TILE_Y][REG_TILE_X] = {0};
    
    // 每个线程计算REG_TILE_Y x REG_TILE_X个输出
    #pragma unroll
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        #pragma unroll
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            float filter_val = d_Filter[(fy + FILTER_RADIUS) * FILTER_SIZE + 
                                       (fx + FILTER_RADIUS)];
            
            #pragma unroll
            for(int ty = 0; ty < REG_TILE_Y; ty++) {
                #pragma unroll
                for(int tx = 0; tx < REG_TILE_X; tx++) {
                    int x = base_x + tx;
                    int y = base_y + ty;
                    int in_x = x + fx;
                    int in_y = y + fy;
                    
                    if(x < width && y < height &&
                       in_x >= 0 && in_x < width && 
                       in_y >= 0 && in_y < height) {
                        results[ty][tx] += input[in_y * width + in_x] * filter_val;
                    }
                }
            }
        }
    }
    
    // 写回结果
    #pragma unroll
    for(int ty = 0; ty < REG_TILE_Y; ty++) {
        #pragma unroll
        for(int tx = 0; tx < REG_TILE_X; tx++) {
            int x = base_x + tx;
            int y = base_y + ty;
            if(x < width && y < height) {
                output[y * width + x] = results[ty][tx];
            }
        }
    }
}

// ============================================================================
// 性能测试框架
// ============================================================================
typedef struct {
    const char* name;
    float time_ms;
    float bandwidth_gb;
    float gflops;
    bool correct;
} BenchmarkResult;

// 性能测试函数
float benchmark_kernel(void (*kernel)(float*, float*, int, int), 
                      float* d_input, float* d_output,
                      int width, int height, int iterations = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<dim3((width + TILE_SIZE - 1) / TILE_SIZE, 
                  (height + TILE_SIZE - 1) / TILE_SIZE),
             dim3(TILE_SIZE, TILE_SIZE)>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    
    // 实际测试
    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        kernel<<<dim3((width + TILE_SIZE - 1) / TILE_SIZE, 
                      (height + TILE_SIZE - 1) / TILE_SIZE),
                 dim3(TILE_SIZE, TILE_SIZE)>>>(d_input, d_output, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / iterations;
}

// 验证结果
bool verify_output(float* h_ref, float* h_test, int size, float tolerance = 1e-4f) {
    for(int i = 0; i < size; i++) {
        if(fabs(h_ref[i] - h_test[i]) > tolerance) {
            printf("  Mismatch at %d: ref=%f, test=%f\n", i, h_ref[i], h_test[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// 主测试函数
// ============================================================================
void run_comprehensive_test(int width, int height) {
    printf("\n========================================\n");
    printf("Testing size: %d x %d\n", width, height);
    printf("========================================\n");
    
    int size = width * height;
    size_t bytes = size * sizeof(float);
    
    // 分配主机内存
    float *h_input = (float*)malloc(bytes);
    float *h_output_ref = (float*)malloc(bytes);
    float *h_output_test = (float*)malloc(bytes);
    float *h_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 初始化数据
    for(int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    for(int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        h_filter[i] = 1.0f / (FILTER_SIZE * FILTER_SIZE);
    }
    
    // 计算CPU参考结果
    printf("Computing CPU reference...\n");
    convolution_cpu_reference(h_input, h_output_ref, h_filter, width, height);
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 创建纹理对象
    cudaTextureObject_t texObj = 0;
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = bytes;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    // 测试结果数组
    BenchmarkResult results[8];
    
    // 配置
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("\nRunning benchmarks...\n");
    printf("%-30s %10s %12s %10s %10s\n", 
           "Strategy", "Time(ms)", "Bandwidth", "GFLOPS", "Status");
    printf("%-30s %10s %12s %10s %10s\n", 
           "--------", "--------", "---------", "------", "------");
    
    // 理论计算
    long long total_ops = (long long)size * FILTER_SIZE * FILTER_SIZE * 2;  // MAD = 2 ops
    long long bytes_accessed = (long long)size * FILTER_SIZE * FILTER_SIZE * sizeof(float) + 
                              size * sizeof(float);  // input reads + output writes
    
    // 1. Direct Naive
    {
        float time_ms = benchmark_kernel(
            (void (*)(float*, float*, int, int))convolution2D_direct_naive,
            d_input, d_output, width, height);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "Direct Global (Naive)", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
    }
    
    // 2. L1 Cache (__ldg)
    {
        float time_ms = benchmark_kernel(
            (void (*)(float*, float*, int, int))convolution2D_L1_cache,
            d_input, d_output, width, height);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "L1 Cache (__ldg)", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
    }
    
    // 3. L2 Prefetch
    {
        float time_ms = benchmark_kernel(
            (void (*)(float*, float*, int, int))convolution2D_L2_prefetch,
            d_input, d_output, width, height);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "L2 Cache (Prefetch)", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
    }
    
    // 4. Texture Memory
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        convolution2D_texture<<<gridDim, blockDim>>>(texObj, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "Texture Memory", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // 5. Shared Memory (Full)
    {
        float time_ms = benchmark_kernel(
            (void (*)(float*, float*, int, int))convolution2D_shared_full,
            d_input, d_output, width, height);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "Shared Memory (Full Halo)", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
    }
    
    // 6. Shared + L1/L2 Hybrid
    {
        float time_ms = benchmark_kernel(
            (void (*)(float*, float*, int, int))convolution2D_shared_hybrid,
            d_input, d_output, width, height);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "Shared + Cache (Hybrid)", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
    }
    
    // 7. Register Blocking
    {
        dim3 regBlockDim(TILE_SIZE/REG_TILE_X, TILE_SIZE/REG_TILE_Y);
        dim3 regGridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                       (height + TILE_SIZE - 1) / TILE_SIZE);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        convolution2D_register_blocking<<<regGridDim, regBlockDim>>>(
            d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        cudaMemcpy(h_output_test, d_output, bytes, cudaMemcpyDeviceToHost);
        bool correct = verify_output(h_output_ref, h_output_test, size);
        
        float bandwidth = (bytes_accessed / 1e9) / (time_ms / 1000.0f);
        float gflops = (total_ops / 1e9) / (time_ms / 1000.0f);
        
        printf("%-30s %10.3f %10.2f GB/s %10.2f %10s\n", 
               "Register Blocking", time_ms, bandwidth, gflops, 
               correct ? "PASS" : "FAIL");
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // 清理
    cudaDestroyTextureObject(texObj);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_ref);
    free(h_output_test);
    free(h_filter);
}

// ============================================================================
// 内存访问模式分析
// ============================================================================
void analyze_memory_patterns() {
    printf("\n================================================\n");
    printf("Memory Access Pattern Analysis\n");
    printf("================================================\n\n");
    
    printf("Filter Size: %d x %d (radius = %d)\n", FILTER_SIZE, FILTER_SIZE, FILTER_RADIUS);
    printf("Tile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Block Size (with halo): %d x %d\n\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // 分析每种策略的内存访问特征
    printf("%-25s %15s %15s %15s %15s\n", 
           "Strategy", "Global Reads", "Shared Mem", "Cache Usage", "Complexity");
    printf("%-25s %15s %15s %15s %15s\n",
           "------------------------", "--------------", "--------------", 
           "--------------", "--------------");
    
    int tile_elements = TILE_SIZE * TILE_SIZE;
    int block_elements = BLOCK_SIZE * BLOCK_SIZE;
    int filter_elements = FILTER_SIZE * FILTER_SIZE;
    
    // Direct Naive
    printf("%-25s %15s %15s %15s %15s\n",
           "Direct Naive", 
           "Very High", 
           "0 B",
           "None",
           "Simple");
    
    // L1 Cache
    printf("%-25s %15s %15s %15s %15s\n",
           "L1 Cache (__ldg)", 
           "High", 
           "0 B",
           "L1 only",
           "Simple");
    
    // L2 Prefetch
    printf("%-25s %15s %15s %15s %15s\n",
           "L2 Prefetch", 
           "High", 
           "0 B",
           "L2 optimized",
           "Moderate");
    
    // Texture
    printf("%-25s %15s %15s %15s %15s\n",
           "Texture Memory", 
           "Medium", 
           "0 B",
           "Texture Cache",
           "Simple");
    
    // Shared Full
    char shared_full[32];
    sprintf(shared_full, "%d B", block_elements * 4);
    printf("%-25s %15s %15s %15s %15s\n",
           "Shared (Full)", 
           "Low", 
           shared_full,
           "None",
           "Complex");
    
    // Shared Hybrid
    char shared_hybrid[32];
    sprintf(shared_hybrid, "%d B", tile_elements * 4);
    printf("%-25s %15s %15s %15s %15s\n",
           "Shared + Cache", 
           "Medium", 
           shared_hybrid,
           "L1/L2 for halo",
           "Moderate");
    
    // Register Blocking
    printf("%-25s %15s %15s %15s %15s\n",
           "Register Blocking", 
           "Medium-High", 
           "0 B",
           "Register reuse",
           "Complex");
    
    // 计算理论值
    printf("\n\nTheoretical Analysis:\n");
    printf("---------------------\n");
    
    float halo_overhead = 100.0f * (block_elements - tile_elements) / block_elements;
    printf("Halo Overhead: %.1f%% extra elements per tile\n", halo_overhead);
    
    int reuse_interior = filter_elements;  // 每个内部元素被filter_size个输出使用
    int reuse_halo = 2;  // Halo元素平均被2个tile共享
    printf("Data Reuse Factor:\n");
    printf("  - Interior elements: %dx\n", reuse_interior);
    printf("  - Halo elements: ~%dx (average)\n", reuse_halo);
    
    float shared_bandwidth_reduction = (float)tile_elements / block_elements;
    printf("Shared Memory Bandwidth Reduction: %.1fx\n", 1.0f / shared_bandwidth_reduction);
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char **argv) {
    printf("================================================\n");
    printf("CUDA Convolution: Complete Strategy Comparison\n");
    printf("================================================\n");
    
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    printf("Memory Bandwidth: %.2f GB/s\n\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    // 内存访问模式分析
    analyze_memory_patterns();
    
    // 运行不同大小的测试
    for(int i = 0; i < NUM_TEST_SIZES; i++) {
        int size = TEST_SIZES[i];
        run_comprehensive_test(size, size);
    }
    
    printf("\n================================================\n");
    printf("All tests completed!\n");
    printf("================================================\n");
    
    return 0;
}
