#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include "ops/vector_add.cuh"

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
    // vectorAddCPU(h_a, h_b, h_c, array_size);
    ASSERT_EQ(err, cudaSuccess) << "CUDA错误: " << cudaGetErrorString(err);

    // 验证结果
    for (int i = 0; i < array_size; i++) {
        EXPECT_NEAR(h_c[i], expected[i], epsilon) 
            << "位置 " << i 
            << " 的结果不匹配。期望: " << expected[i] 
            << ", 实际: " << h_c[i];
    }
}

// 测试性能
TEST_F(VectorAddTest, Performance) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热GPU
    vectorAdd(h_a, h_b, h_c, array_size);

    // 计时测试
    cudaEventRecord(start);
    cudaError_t err = vectorAdd(h_a, h_b, h_c, array_size);
    cudaEventRecord(stop);
    
    ASSERT_EQ(err, cudaSuccess) << "CUDA错误: " << cudaGetErrorString(err);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算和输出性能指标
    float bandwidth = (3 * array_size * sizeof(float)) / (milliseconds * 1e6);
    std::cout << "性能测试结果：" << std::endl;
    std::cout << "处理 " << array_size << " 个元素耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "带宽: " << bandwidth << " GB/s" << std::endl;

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 性能断言（示例阈值，需要根据实际硬件调整）
    EXPECT_LT(milliseconds, 10.0f) << "性能低于预期";
}

// 测试边界情况
TEST_F(VectorAddTest, ZeroSize) {
    cudaError_t err = vectorAdd(h_a, h_b, h_c, 0);
    EXPECT_EQ(err, cudaSuccess) << "处理空数组应该成功";
}

// 测试大数据量
TEST_F(VectorAddTest, LargeArray) {
    const int large_size = 10000;  // 1千万个元素
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 