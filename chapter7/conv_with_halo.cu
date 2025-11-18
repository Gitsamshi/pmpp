// CUDA卷积算法实现 - Halo Cells处理
// 包含两种策略：1) 加载halo到共享内存 2) 使用缓存处理halo

#include <stdio.h>
#include <cuda_runtime.h>

// 常量定义
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + 2 * FILTER_RADIUS)

// 将卷积核放在常量内存中
__constant__ float d_Filter[FILTER_SIZE * FILTER_SIZE];

//---------------------------------------------------------------------------
// 策略1: 将halo cells加载到共享内存
// 这个版本将整个输入块（包括halo）都加载到共享内存
//---------------------------------------------------------------------------
__global__ void convolution2D_with_halo_in_shared(
    float *d_Input, 
    float *d_Output,
    int width, 
    int height)
{
    // 声明共享内存 - 大小包含halo区域
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    
    // 计算线程在输出中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算输出元素的全局坐标
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // 计算输入块的起始位置（考虑halo）
    int in_x_start = blockIdx.x * TILE_SIZE - FILTER_RADIUS;
    int in_y_start = blockIdx.y * TILE_SIZE - FILTER_RADIUS;
    
    // 每个线程可能需要加载多个元素以覆盖整个输入块
    // 这是因为输入块（含halo）比线程块大
    int n_loads_x = (BLOCK_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    int n_loads_y = (BLOCK_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    
    // 协作加载输入块到共享内存
    for(int j = 0; j < n_loads_y; j++) {
        for(int i = 0; i < n_loads_x; i++) {
            int tile_x = tx + i * TILE_SIZE;
            int tile_y = ty + j * TILE_SIZE;
            
            if(tile_x < BLOCK_SIZE && tile_y < BLOCK_SIZE) {
                int in_x = in_x_start + tile_x;
                int in_y = in_y_start + tile_y;
                
                // 边界检查 - 处理ghost cells
                if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    tile[tile_y][tile_x] = d_Input[in_y * width + in_x];
                } else {
                    tile[tile_y][tile_x] = 0.0f; // Ghost cells设为0
                }
            }
        }
    }
    
    __syncthreads();
    
    // 计算卷积输出
    if(out_x < width && out_y < height && tx < TILE_SIZE && ty < TILE_SIZE) {
        float value = 0.0f;
        
        // 遍历卷积核
        for(int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for(int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                // 在共享内存中的位置（已包含halo偏移）
                int tile_row = ty + fRow;
                int tile_col = tx + fCol;
                
                value += d_Filter[fRow * FILTER_SIZE + fCol] * 
                         tile[tile_row][tile_col];
            }
        }
        
        d_Output[out_y * width + out_x] = value;
    }
}

//---------------------------------------------------------------------------
// 策略2: 使用缓存处理halo cells
// 只将内部元素加载到共享内存，halo cells通过L1/L2缓存访问
//---------------------------------------------------------------------------
__global__ void convolution2D_with_cached_halo(
    float *d_Input,
    float *d_Output,
    int width,
    int height)
{
    // 共享内存只存储内部元素（不包含halo）
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 输出位置
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // 加载内部元素到共享内存
    if(out_x < width && out_y < height) {
        tile[ty][tx] = d_Input[out_y * width + out_x];
    } else {
        tile[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // 计算卷积
    if(out_x < width && out_y < height) {
        float value = 0.0f;
        
        for(int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
            for(int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
                // 计算输入坐标
                int in_x = out_x + fCol;
                int in_y = out_y + fRow;
                
                // 判断是否在tile内部
                int tile_x = tx + fCol;
                int tile_y = ty + fRow;
                
                float input_value;
                
                // 决定从哪里读取数据
                if(tile_x >= 0 && tile_x < TILE_SIZE && 
                   tile_y >= 0 && tile_y < TILE_SIZE) {
                    // 内部元素 - 从共享内存读取
                    input_value = tile[tile_y][tile_x];
                } else if(in_x >= 0 && in_x < width && 
                         in_y >= 0 && in_y < height) {
                    // Halo cells - 从全局内存读取（依赖缓存）
                    input_value = d_Input[in_y * width + in_x];
                } else {
                    // Ghost cells - 边界外
                    input_value = 0.0f;
                }
                
                // 应用卷积核
                int filter_idx = (fRow + FILTER_RADIUS) * FILTER_SIZE + 
                                (fCol + FILTER_RADIUS);
                value += d_Filter[filter_idx] * input_value;
            }
        }
        
        d_Output[out_y * width + out_x] = value;
    }
}

//---------------------------------------------------------------------------
// 1D卷积示例 - 展示halo概念
//---------------------------------------------------------------------------
__global__ void convolution1D_with_halo(
    float *d_Input,
    float *d_Output,
    float *d_Filter1D,
    int size)
{
    // 共享内存包含halo
    extern __shared__ float s_Input[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 计算这个块需要的输入范围
    int inputStart = blockIdx.x * blockDim.x - FILTER_RADIUS;
    int inputEnd = (blockIdx.x + 1) * blockDim.x + FILTER_RADIUS - 1;
    
    // 协作加载数据到共享内存
    // 每个线程可能加载多个元素
    for(int i = tid; i <= (inputEnd - inputStart); i += blockDim.x) {
        int inputIdx = inputStart + i;
        
        // 边界检查
        if(inputIdx >= 0 && inputIdx < size) {
            s_Input[i] = d_Input[inputIdx];
        } else {
            s_Input[i] = 0.0f; // Ghost cells
        }
    }
    
    __syncthreads();
    
    // 执行卷积
    if(gid < size) {
        float value = 0.0f;
        
        for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            // 在共享内存中的索引（已经包含了偏移）
            int sharedIdx = tid + FILTER_RADIUS + i;
            value += s_Input[sharedIdx] * d_Filter1D[i + FILTER_RADIUS];
        }
        
        d_Output[gid] = value;
    }
}

//---------------------------------------------------------------------------
// 主机端辅助函数
//---------------------------------------------------------------------------

// 初始化数据
void initializeData(float *data, int size) {
    for(int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 10);
    }
}

// 初始化高斯滤波器
void initializeGaussianFilter(float *filter, int radius) {
    int size = 2 * radius + 1;
    float sum = 0.0f;
    float sigma = 1.0f;
    
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            filter[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    
    // 归一化
    for(int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

// 主函数
int main() {
    // 图像尺寸
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    
    // 分配主机内存
    float *h_Input = (float*)malloc(size * sizeof(float));
    float *h_Output1 = (float*)malloc(size * sizeof(float));
    float *h_Output2 = (float*)malloc(size * sizeof(float));
    float *h_Filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 初始化数据
    initializeData(h_Input, size);
    initializeGaussianFilter(h_Filter, FILTER_RADIUS);
    
    // 分配设备内存
    float *d_Input, *d_Output1, *d_Output2;
    cudaMalloc(&d_Input, size * sizeof(float));
    cudaMalloc(&d_Output1, size * sizeof(float));
    cudaMalloc(&d_Output2, size * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_Input, h_Input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Filter, h_Filter, 
                       FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // 定义网格和块维度
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 测试策略1：Halo in Shared Memory
    printf("执行策略1: Halo cells in shared memory...\n");
    cudaEventRecord(start);
    
    convolution2D_with_halo_in_shared<<<gridDim, blockDim>>>(
        d_Input, d_Output1, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    printf("策略1执行时间: %.3f ms\n", time1);
    
    // 测试策略2：Cached Halo
    printf("\n执行策略2: Halo cells with caching...\n");
    cudaEventRecord(start);
    
    convolution2D_with_cached_halo<<<gridDim, blockDim>>>(
        d_Input, d_Output2, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    printf("策略2执行时间: %.3f ms\n", time2);
    
    // 复制结果回主机
    cudaMemcpy(h_Output1, d_Output1, size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Output2, d_Output2, size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // 验证两种方法的结果是否一致
    float maxDiff = 0.0f;
    for(int i = 0; i < size; i++) {
        float diff = fabs(h_Output1[i] - h_Output2[i]);
        if(diff > maxDiff) maxDiff = diff;
    }
    printf("\n两种策略的最大差异: %e\n", maxDiff);
    
    // 分析性能差异
    printf("\n性能分析:\n");
    printf("加速比 (策略1/策略2): %.2fx\n", time2/time1);
    
    // 计算理论分析
    int halosPerTile = (TILE_SIZE + 2 * FILTER_RADIUS) * 
                       (TILE_SIZE + 2 * FILTER_RADIUS) - 
                       TILE_SIZE * TILE_SIZE;
    printf("\n理论分析:\n");
    printf("每个tile的内部元素: %d\n", TILE_SIZE * TILE_SIZE);
    printf("每个tile的halo元素: %d\n", halosPerTile);
    printf("Halo占比: %.1f%%\n", 
           100.0f * halosPerTile / (TILE_SIZE * TILE_SIZE + halosPerTile));
    
    // 清理
    cudaFree(d_Input);
    cudaFree(d_Output1);
    cudaFree(d_Output2);
    free(h_Input);
    free(h_Output1);
    free(h_Output2);
    free(h_Filter);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
