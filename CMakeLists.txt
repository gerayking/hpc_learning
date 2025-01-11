cmake_minimum_required(VERSION 3.10)
project(hpc_learning CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构（根据你的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 70)

# 查找GTest包
set(GTest_DIR "/usr/local/lib/cmake/GTest")
find_package(GTest REQUIRED)
message(STATUS "GTest include directories: ${GTEST_INCLUDE_DIRS}")
message(STATUS "GTest libraries: ${GTEST_LIBRARIES}")

# 添加include目录
link_directories("/usr/local/lib")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ops)

# 添加CUDA源文件
add_library(vector_add STATIC
    src/ops/cuda/vector_add.cu
)

# 设置编译选项以生成PTX和SASS
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70
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

# vector_add_profile
add_executable(vector_add_profile
    tests/ops/cuda/vector_add_profile.cu
)
target_link_libraries(vector_add_profile
    PRIVATE
    vector_add
)
set_target_properties(vector_add_profile PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70
)


#reduce
add_library(reduce STATIC
    src/ops/cuda/reduce.cu
)

# reduce_profile
add_executable(reduce_profile
    tests/ops/cuda/reduce_profile.cu
)
target_link_libraries(reduce_profile
    PRIVATE
    reduce
)

# reduce_test
add_executable(reduce_test
    tests/ops/cuda/reduce_test.cu
)

target_link_libraries(reduce_test
    PRIVATE
    reduce
)

add_custom_target(profile
    COMMAND vector_add_profile
    COMMAND reduce_profile
    DEPENDS vector_add_profile reduce_profile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all profile executables"
)

# 启用测试
enable_testing()
add_test(NAME vector_add_test COMMAND vector_add_test) 
add_test(NAME reduce_test COMMAND reduce_test) 