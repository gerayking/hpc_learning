#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include "ops/vector_add.h"

void test_vector_add_kernel(int size,int GridSize,int BlockSize, cudaEvent_t start, cudaEvent_t stop, void (*kernel)(float*, float*, float*, int), std::string kernel_name) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    // 初始化数据
    std::vector<float> h_a(size, 1.0f);
    std::vector<float> h_b(size, 2.0f);
    for(int i = 0; i < size; i++){
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
    cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    // 计时测试
    cudaEventRecord(start);
    kernel<<<GridSize, BlockSize>>>(d_a, d_b, d_c, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::vector<float> h_c(size);
    std::vector<float> h_d(size);
    cudaMemcpy(h_c.data(), d_c, 100 * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < 100; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        std::cerr << "v0 结果不正确，数组大小: " << size << std::endl;
    }
    float bandwidth = (3 * size * sizeof(float)) / (milliseconds * 1e6);

    printf("%8s\t%9d\t%8.3f\t%8.2f\n", kernel_name.c_str(), size, milliseconds, bandwidth);
    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void test_vector_add_kernel_v3(int size,int GridSize,int BlockSize, cudaEvent_t start, cudaEvent_t stop, void (*kernel)(float*, float*, float*, int, int), std::string kernel_name, int stride) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    // 初始化数据
    std::vector<float> h_a(size, 1.0f);
    std::vector<float> h_b(size, 2.0f);
    for(int i = 0; i < size; i++){
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
    cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    // 计时测试
    cudaEventRecord(start);
    kernel<<<GridSize, BlockSize>>>(d_a, d_b, d_c, size, stride);
    cudaEventRecord(stop);
    cudaError_t err = cudaGetLastError();  // 获取最后一个错误
    if (err != cudaSuccess) {  // 检查错误
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::vector<float> h_c(size);
    std::vector<float> h_d(size);
    cudaMemcpy(h_c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            printf("%d: %f = %f + %f\n", i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    if (!correct) {
        std::cerr << kernel_name << " 结果不正确，数组大小: " << size << std::endl;
    }
    float bandwidth = (3 * size * sizeof(float)) / (milliseconds * 1e6);

    printf("%8s\t%9d\t%9d\t%9d\t%8.3f\t%8.2f\n", kernel_name.c_str(), GridSize, BlockSize, size, milliseconds, bandwidth);
    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char **argv) {
    std::vector<int> test_sizes = {
        1024 * 16,        // 100K
        1024 * 1024,       // 1M
        1024 * 1024 * 12,      // 12MB
        1024 * 1024 * 128,     // 128MB
    };

    cudaEvent_t start, stop, start_v1, stop_v1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_v1);
    cudaEventCreate(&stop_v1);

    std::cout << "\n内核性能测试结果：" << std::endl;
    std::cout << "版本名称\tGridSize\tBlockSize\t数组大小\t耗时(ms)\t带宽(GB/s)" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        int BlockSize = 256;
        int GridSize = (size + 255) / BlockSize;
        test_vector_add_kernel(size, GridSize, BlockSize, start, stop, vector_add_kernel_v0, "v0");
        for(int i = 0; i < 10; i++){
            GridSize = (size + 255) / BlockSize / (i + 1);
        }
        GridSize = (size + 255) / BlockSize / 4;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v1", 1);
        GridSize = (size + 255) / BlockSize / 8;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v2", 2);
        GridSize = (size + 255) / BlockSize / 16;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v3", 4);
        GridSize = (size + 255) / BlockSize / 32;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v4", 8);
        GridSize = (size + 255) / BlockSize / 64;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v5", 16);
        GridSize = (size + 255) / BlockSize / 128;
        test_vector_add_kernel_v3(size, GridSize, BlockSize, start, stop, vector_add_kernel_v3, "v6", 32);
        // ... existing code for v1 ...
    }

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}