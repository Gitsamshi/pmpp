// CUDA卷积算法 - Halo Cells处理详解
// 这是一个教学示例，展示如何处理halo cells

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// 配置参数
// ============================================================================
#define FILTER_RADIUS 1        // 卷积核半径（3x3卷积核）
#define FILTER_SIZE 3          // 卷积核大小 = 2 * RADIUS + 1
#define TILE_SIZE 4           // 每个线程块处理的输出tile大小
#define BLOCK_SIZE 6          // 包含halo的输入块大小 = TILE_SIZE + 2*RADIUS

// 将卷积核放在常量内存中（所有线程共享，缓存友好）
__constant__ float d_Filter[FILTER_SIZE * FILTER_SIZE];

// ============================================================================
// 核心算法实现
// ============================================================================

/**
 * 基础版本：无优化的2D卷积
 * 每个线程计算一个输出元素，直接从全局内存读取
 */
__global__ void convolution2D_basic(
    float *input, 
    float *output,
    int width, 
    int height)
{
    // 计算当前线程负责的输出位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查
    if(x >= width || y >= height) return;
    
    float result = 0.0f;
    
    // 应用卷积核
    for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
        for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
            // 计算输入位置
            int in_x = x + fx;
            int in_y = y + fy;
            
            // 处理边界（ghost cells）
            if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                int filter_idx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
                result += input[in_y * width + in_x] * d_Filter[filter_idx];
            }
            // 如果超出边界，隐式地当作0处理（不累加）
        }
    }
    
    output[y * width + x] = result;
}

/**
 * 优化版本1：将halo cells加载到共享内存
 * 
 * 内存布局示例（TILE_SIZE=4, FILTER_RADIUS=1）：
 * 
 * 共享内存块（6x6）：
 * +---+---+---+---+---+---+
 * | H | H | H | H | H | H |  <- Halo顶部
 * +---+---+---+---+---+---+
 * | H | I | I | I | I | H |  <- H=Halo, I=Interior
 * +---+---+---+---+---+---+
 * | H | I | I | I | I | H |
 * +---+---+---+---+---+---+
 * | H | I | I | I | I | H |
 * +---+---+---+---+---+---+
 * | H | I | I | I | I | H |
 * +---+---+---+---+---+---+
 * | H | H | H | H | H | H |  <- Halo底部
 * +---+---+---+---+---+---+
 */
__global__ void convolution2D_tiled_with_halo(
    float *input,
    float *output,
    int width,
    int height)
{
    // 分配共享内存（包含halo）
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    
    // 线程在块内的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 输出元素的全局位置
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // ====================================================================
    // 第1步：协作加载输入数据到共享内存（包括halo）
    // ====================================================================
    
    // 计算输入块的起始位置（考虑halo偏移）
    int in_block_start_x = blockIdx.x * TILE_SIZE - FILTER_RADIUS;
    int in_block_start_y = blockIdx.y * TILE_SIZE - FILTER_RADIUS;
    
    // 由于输入块(6x6)比线程块(4x4)大，每个线程可能需要加载多个元素
    // 策略：使用循环让每个线程加载多个元素
    
    #pragma unroll
    for(int load_y = 0; load_y < BLOCK_SIZE; load_y += TILE_SIZE) {
        #pragma unroll
        for(int load_x = 0; load_x < BLOCK_SIZE; load_x += TILE_SIZE) {
            // 当前加载的共享内存位置
            int tile_y = ty + load_y;
            int tile_x = tx + load_x;
            
            // 确保不越界共享内存
            if(tile_x < BLOCK_SIZE && tile_y < BLOCK_SIZE) {
                // 对应的全局内存位置
                int global_x = in_block_start_x + tile_x;
                int global_y = in_block_start_y + tile_y;
                
                // 处理ghost cells
                if(global_x >= 0 && global_x < width && 
                   global_y >= 0 && global_y < height) {
                    tile[tile_y][tile_x] = input[global_y * width + global_x];
                } else {
                    tile[tile_y][tile_x] = 0.0f;  // Ghost cell设为0
                }
            }
        }
    }
    
    // 同步：确保所有线程完成数据加载
    __syncthreads();
    
    // ====================================================================
    // 第2步：从共享内存计算输出
    // ====================================================================
    
    // 只有负责有效输出的线程才计算
    if(out_x < width && out_y < height && tx < TILE_SIZE && ty < TILE_SIZE) {
        float result = 0.0f;
        
        // 应用卷积核
        #pragma unroll
        for(int fy = 0; fy < FILTER_SIZE; fy++) {
            #pragma unroll
            for(int fx = 0; fx < FILTER_SIZE; fx++) {
                // 在共享内存中的位置
                // 注意：由于halo的存在，需要偏移
                int smem_y = ty + fy;  // ty已经在正确的位置
                int smem_x = tx + fx;  // tx已经在正确的位置
                
                result += tile[smem_y][smem_x] * d_Filter[fy * FILTER_SIZE + fx];
            }
        }
        
        output[out_y * width + out_x] = result;
    }
}

/**
 * 优化版本2：使用缓存处理halo cells
 * 只将内部元素加载到共享内存，halo通过L1/L2缓存访问
 */
__global__ void convolution2D_tiled_cached_halo(
    float *input,
    float *output,
    int width,
    int height)
{
    // 共享内存只存储内部元素
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // ====================================================================
    // 第1步：只加载内部元素到共享内存
    // ====================================================================
    if(out_x < width && out_y < height) {
        tile[ty][tx] = input[out_y * width + out_x];
    } else {
        tile[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // ====================================================================
    // 第2步：计算输出（混合访问共享内存和全局内存）
    // ====================================================================
    if(out_x < width && out_y < height) {
        float result = 0.0f;
        
        #pragma unroll
        for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
            #pragma unroll
            for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
                int in_x = out_x + fx;
                int in_y = out_y + fy;
                
                // 检查是否在tile内部
                int tile_x = tx + fx;
                int tile_y = ty + fy;
                
                float value;
                
                // 三种情况的处理
                if(tile_x >= 0 && tile_x < TILE_SIZE && 
                   tile_y >= 0 && tile_y < TILE_SIZE) {
                    // 情况1：内部元素 - 从共享内存读取（快）
                    value = tile[tile_y][tile_x];
                } else if(in_x >= 0 && in_x < width && 
                         in_y >= 0 && in_y < height) {
                    // 情况2：Halo cells - 从全局内存读取（依赖缓存）
                    value = input[in_y * width + in_x];
                } else {
                    // 情况3：Ghost cells - 边界外
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
// 辅助函数
// ============================================================================

/**
 * 初始化输入数据（简单的递增序列）
 */
void initializeInput(float *data, int width, int height) {
    for(int i = 0; i < width * height; i++) {
        data[i] = i % 256;  // 防止数值过大
    }
}

/**
 * 初始化卷积核（高斯滤波器）
 */
void initializeGaussianFilter(float *filter) {
    float kernel[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
    
    for(int i = 0; i < 9; i++) {
        filter[i] = kernel[i];
    }
}

/**
 * 验证两个输出是否相同
 */
bool verifyResults(float *result1, float *result2, int size) {
    const float epsilon = 1e-5;
    for(int i = 0; i < size; i++) {
        if(fabs(result1[i] - result2[i]) > epsilon) {
            printf("Mismatch at position %d: %f vs %f\n", 
                   i, result1[i], result2[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("================================================\n");
    printf("CUDA Convolution with Halo Cells Demonstration\n");
    printf("================================================\n\n");
    
    // 设置问题规模
    const int width = 64;
    const int height = 64;
    const int size = width * height;
    
    printf("Configuration:\n");
    printf("- Image size: %d x %d\n", width, height);
    printf("- Filter size: %d x %d (radius=%d)\n", 
           FILTER_SIZE, FILTER_SIZE, FILTER_RADIUS);
    printf("- Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("- Block size with halo: %d x %d\n\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // 分配主机内存
    size_t bytes = size * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output_basic = (float*)malloc(bytes);
    float *h_output_tiled = (float*)malloc(bytes);
    float *h_output_cached = (float*)malloc(bytes);
    float *h_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 初始化数据
    initializeInput(h_input, width, height);
    initializeGaussianFilter(h_filter);
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 设置执行配置
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Execution configuration:\n");
    printf("- Grid: %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("- Block: %d x %d threads\n", blockDim.x, blockDim.y);
    printf("- Total threads: %d\n\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // ====================================================================
    // 执行不同版本的核函数
    // ====================================================================
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    // 1. 基础版本
    printf("Running basic convolution...\n");
    cudaEventRecord(start);
    convolution2D_basic<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("- Time: %.3f ms\n", milliseconds);
    cudaMemcpy(h_output_basic, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // 2. Tiled with halo in shared memory
    printf("\nRunning tiled convolution (halo in shared memory)...\n");
    cudaEventRecord(start);
    convolution2D_tiled_with_halo<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("- Time: %.3f ms\n", milliseconds);
    cudaMemcpy(h_output_tiled, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // 3. Tiled with cached halo
    printf("\nRunning tiled convolution (cached halo)...\n");
    cudaEventRecord(start);
    convolution2D_tiled_cached_halo<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("- Time: %.3f ms\n", milliseconds);
    cudaMemcpy(h_output_cached, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // ====================================================================
    // 验证结果
    // ====================================================================
    printf("\nVerifying results...\n");
    bool tiled_correct = verifyResults(h_output_basic, h_output_tiled, size);
    bool cached_correct = verifyResults(h_output_basic, h_output_cached, size);
    
    printf("- Tiled version: %s\n", tiled_correct ? "PASSED" : "FAILED");
    printf("- Cached version: %s\n", cached_correct ? "PASSED" : "FAILED");
    
    // ====================================================================
    // 分析halo开销
    // ====================================================================
    printf("\nHalo overhead analysis:\n");
    
    int total_tiles = gridDim.x * gridDim.y;
    int interior_per_tile = TILE_SIZE * TILE_SIZE;
    int total_per_tile = BLOCK_SIZE * BLOCK_SIZE;
    int halo_per_tile = total_per_tile - interior_per_tile;
    
    printf("- Number of tiles: %d\n", total_tiles);
    printf("- Interior elements per tile: %d\n", interior_per_tile);
    printf("- Halo elements per tile: %d\n", halo_per_tile);
    printf("- Total elements per tile: %d\n", total_per_tile);
    printf("- Halo overhead: %.1f%%\n", 100.0f * halo_per_tile / total_per_tile);
    
    // 计算内存访问统计
    int total_interior = interior_per_tile * total_tiles;
    int total_with_halo = total_per_tile * total_tiles;
    float reuse_factor = (float)total_with_halo / size;
    
    printf("\nMemory access statistics:\n");
    printf("- Unique elements in image: %d\n", size);
    printf("- Total elements accessed (with halo): %d\n", total_with_halo);
    printf("- Average reuse factor: %.2fx\n", reuse_factor);
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_basic);
    free(h_output_tiled);
    free(h_output_cached);
    free(h_filter);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n================================================\n");
    printf("Demonstration complete!\n");
    printf("================================================\n");
    
    return 0;
}
