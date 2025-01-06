#include <cuda_runtime.h>
#include <random>
#include <stdio.h>

// CUDA kernel函数 - 在GPU上执行向量加法
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // 计算当前线程处理的元素索引
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保线程索引不超过数组边界
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机端调用函数
cudaError_t vectorAdd(const float* h_a, const float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    cudaError_t err = cudaSuccess;

    // 在GPU上分配内存
    err = cudaMalloc((void**)&d_a, n * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc((void**)&d_b, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return err;
    }
    
    err = cudaMalloc((void**)&d_c, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return err;
    }

    // 将输入数据从主机复制到设备
    err = cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    
    err = cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 配置kernel启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 启动kernel
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // 检查kernel执行错误
    err = cudaGetLastError();

    // 将结果从设备复制回主机
    err = cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    return err;
}

int main() {
    const int N = 1024 * 1024;  // 测试数组大小
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    // 分配主机内存并初始化数据
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // 初始化输入数组
    for (int i = 0; i < N; i++) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
        h_c[i] = h_a[i] + h_b[i];
    }
    
    // 调用GPU向量加法函数
    cudaError_t err = vectorAdd(h_a, h_b, h_c, N);
    
    // 验证结果
    if (err == cudaSuccess) {
        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                correct = false;
                printf("结果错误: h_c[%d] = %f\n", i, h_c[i]);
                break;
            }
        }
        if (correct) {
            printf("向量加法测试通过！ %f = %f + %f\n", h_c[0], h_a[0], h_b[0]);
        }
    } else {
        printf("CUDA错误: %s\n", cudaGetErrorString(err));
    }
    
    // 释放主机内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}