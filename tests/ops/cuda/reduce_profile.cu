#include <cuda_runtime.h>
#include <stdio.h>
#include "ops/reduce.h"
#include <random>   // 添加random头文件

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i]){
            printf("out[%d] = %f, res[%d] = %f\n", i, out[i], i, res[i]);
            return false;
        }
    }
    return true;
}


// 计算带宽的辅助函数
float bandwidth(int bytes, float seconds) {
    return (float)bytes / seconds / 1e9;  // 返回GB/s
}
//timing(reduce_kernel_v1<THREAD_PER_BLOCK>, d_in, d_out, N, block_num, grid, block, "Version 1");
// 计时的辅助函数
void timing(void (*reduce_func)(float*, float*, int), int n,int thread_num, int block_num, dim3 grid, dim3 block, const char* kernel_name, const char *desc) {
    float *h_in, *h_out;
    float *d_in, *d_out;
    h_in = (float*)malloc(n * sizeof(float));
    h_out = (float*)malloc(block_num * sizeof(float));
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, block_num * sizeof(float));
    double sum = 0;
    // 创建随机数生成器
    std::random_device rd;  // 用于获取随机种子
    std::mt19937 gen(rd()); // Mersenne Twister 生成器
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // 均匀分布在[0,1]之间

    // 初始化数组
    for(size_t i = 0; i < n; i++){
        h_in[i] = dis(gen);  // 生成0-1之间的随机浮点数
        sum += h_in[i];
    }
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_out, h_out, block_num * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    // reduce_func<<<grid, block>>>(d_in, d_out, n);
    
    // 计时开始
    cudaEventRecord(start);
    reduce_func<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaEventSynchronize(stop);
    cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    double sum_out = 0;
    for(int i=0;i<block_num;i++){
        sum_out += h_out[i];
    }

    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0;  // 只需要将毫秒转换为秒
    
    // 计算带宽
    int total_bytes = n * sizeof(float);  // 只计算输入数据的读取
    float bw = bandwidth(total_bytes, seconds);
    
    if(fabs(sum - sum_out) > 1e-2){
        printf("check failed %s sum=%f, sum_out=%f, diff=%f\n", kernel_name, sum, sum_out, fabs(sum - sum_out)  );
    }
    printf("%-10s: 耗时=%6.3f ms, 带宽=%6.1f GB/s, %s\n", 
           kernel_name, milliseconds, bw, desc);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    const int THREAD_PER_BLOCK = 256;
    const int N = 128 * 1024 * 1024;  // 32M 元素
    const int block_num = N / THREAD_PER_BLOCK;
    
    // 分配内存
    float *h_in, *d_in, *d_out;
    h_in = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, block_num * sizeof(float));
    
    // 设置grid和block
    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);
    
    // 测试所有版本

    dim3 grid_v3(block_num / 2, 1);
    dim3 block_v3(THREAD_PER_BLOCK, 1);

    
    timing(reduce_kernel_baseline<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num, grid, block, "Baseline", "Baseline");
    timing(reduce_kernel_v1<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num, grid, block, "Version 1", "solve warp divergence");
    timing(reduce_kernel_v2<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num, grid, block, "Version 2", "solve bank conflict");
    timing(reduce_kernel_v3<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num/2, grid_v3, block_v3, "Version 3","parallel compute and load");
    timing(reduce_kernel_v4<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num/2, grid_v3, block_v3, "Version 4","unroll last loop");
    timing(reduce_kernel_v5<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num/2, grid_v3, block_v3, "Version 5", "unroll all loop");
    timing(reduce_kernel_v6<THREAD_PER_BLOCK>, N, THREAD_PER_BLOCK, block_num/2, grid_v3, block_v3, "Version 6", "warp reduce");
    // 清理
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}