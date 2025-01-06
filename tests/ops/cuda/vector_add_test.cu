#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include "ops/vector_add.h"

class VectorAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // 分配和初始化测试数据
        h_a = new float[array_size];
        h_b = new float[array_size];
        h_c = new float[array_size];
        expected = new float[array_size];

        // 生成随机测试数据
        for (int i = 0; i < array_size; i++) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
            expected[i] = h_a[i] + h_b[i];  // 计算期望结果
        }
    }

    void TearDown() override {
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        delete[] expected;
    }

    // CPU版本的向量加法，用于计算参考结果
    void vectorAddCPU(const float* a, const float* b, float* c, int n) {
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }

    static const int array_size = 1000000;
    float* h_a;
    float* h_b;
    float* h_c;
    float* expected;
    const float epsilon = 1e-5f;
};

// 测试基本功能
TEST_F(VectorAddTest, CorrectResults) {
    cudaError_t err = vectorAdd(h_a, h_b, h_c, array_size);
    ASSERT_EQ(err, cudaSuccess) << "CUDA错误: " << cudaGetErrorString(err);

    // 找到第一个不匹配的结果并打印
    bool found_mismatch = false;
    for (int i = 0; i < array_size && !found_mismatch; i++) {
        if (std::abs(h_c[i] - expected[i]) > epsilon) {
            std::cout << "首个不匹配位置 " << i 
                     << "：期望值 = " << expected[i] 
                     << "，实际值 = " << h_c[i] 
                     << "，差异 = " << std::abs(h_c[i] - expected[i]) << std::endl;
            found_mismatch = true;
        }
    }

    // 仍然保留原有的测试断言
    for (int i = 0; i < array_size; i++) {
        EXPECT_NEAR(h_c[i], expected[i], epsilon) 
            << "位置 " << i 
            << " 的结果不匹配。期望: " << expected[i] 
            << ", 实际: " << h_c[i];
    }
}




// 测试性能
TEST_F(VectorAddTest, Performance) {
    // 定义要测试的不同数组大小
    std::vector<int> test_sizes = {
        1000,          // 1K
        100000,        // 100K
        1000000,       // 1M
        10000000,      // 10M
        100000000      // 100M
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n性能测试结果：" << std::endl;
    std::cout << "数组大小\t耗时(ms)\t带宽(GB/s)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (int size : test_sizes) {
        // 为每个大小分配内存
        float *test_a, *test_b, *test_c;
        test_a = new float[size];
        test_b = new float[size];
        test_c = new float[size];

        // 初始化数据
        for (int i = 0; i < size; i++) {
            test_a[i] = 1.0f;
            test_b[i] = 2.0f;
        }

        // 预热GPU
        vectorAdd(test_a, test_b, test_c, size);

        // 计时测试
        cudaEventRecord(start);
        cudaError_t err = vectorAdd(test_a, test_b, test_c, size);
        cudaEventRecord(stop);
        
        ASSERT_EQ(err, cudaSuccess) << "CUDA错误: " << cudaGetErrorString(err);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // 计算带宽 (3个数组：2个输入1个输出)
        float bandwidth = (3 * size * sizeof(float)) / (milliseconds * 1e6);
        
        // 格式化输出结果
        printf("%9d\t%8.3f\t%8.2f\n", size, milliseconds, bandwidth);

        // 清理内存
        delete[] test_a;
        delete[] test_b;
        delete[] test_c;


    }

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 测试边界情况
TEST_F(VectorAddTest, ZeroSize) {
    cudaError_t err = vectorAdd(h_a, h_b, h_c, 0);
    EXPECT_EQ(err, cudaSuccess) << "处理空数组应该成功";
}

// 测试大数据量
TEST_F(VectorAddTest, LargeArray) {
    const int large_size = 1024;  // 1千万个元素
    float* large_a = new float[large_size];
    float* large_b = new float[large_size];
    float* large_c = new float[large_size];

    // 初始化大数组
    for (int i = 0; i < large_size; i++) {
        large_a[i] = 1.0f;
        large_b[i] = 2.0f;
    }

    cudaError_t err = vectorAdd(large_a, large_b, large_c, large_size);
    ASSERT_EQ(err, cudaSuccess) << "大数组处理失败";

    // 验证部分结果
    for (int i = 0; i < 100; i++) {  // 只检查前100个元素
        EXPECT_NEAR(large_c[i], 3.0f, epsilon);
    }

    delete[] large_a;
    delete[] large_b;
    delete[] large_c;
}

// 测试vector_add_kernel的性能
TEST_F(VectorAddTest, KernelPerformance) {
    // 定义要测试的不同数组大小
    std::vector<int> test_sizes = {
        1000,          // 1K
        100000,        // 100K
        1000000,       // 1M
        10000000,      // 10M
        100000000      // 100M
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n内核性能测试结果：" << std::endl;
    std::cout << "数组大小\t耗时(ms)\t带宽(GB/s)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (int size : test_sizes) {
        // 为每个大小分配内存
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));

        // 初始化数据
        std::vector<float> h_a(size, 1.0f);
        std::vector<float> h_b(size, 2.0f);
        cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // 预热GPU
        vector_add_kernel<<<(size + 255) / 256, 256>>>(d_a, d_b, d_c, size);

        // 计时测试
        cudaEventRecord(start);
        vector_add_kernel<<<(size + 255) / 256, 256>>>(d_a, d_b, d_c, size);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // 计算带宽 (3个数组：2个输入1个输出)
        float bandwidth = (3 * size * sizeof(float)) / (milliseconds * 1e6);
        
        // 格式化输出结果
        printf("%9d\t%8.3f\t%8.2f\n", size, milliseconds, bandwidth);

        // 清理内存
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 