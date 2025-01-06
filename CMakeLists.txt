cmake_minimum_required(VERSION 3.10)
project(hpc_learning CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构（根据你的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 75)

# 查找GTest包
find_package(GTest REQUIRED)

# 添加include目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ops)


# 添加CUDA源文件
add_library(vector_add STATIC
    src/ops/cuda/vector_add.cu
)

# 添加测试可执行文件
add_executable(vector_add_test
    tests/ops/cuda/vector_add_test.cu
)

# 链接依赖库
target_link_libraries(vector_add_test
    PRIVATE
    vector_add
    GTest::GTest
    GTest::Main
)

# 启用测试
enable_testing()
add_test(NAME vector_add_test COMMAND vector_add_test) 