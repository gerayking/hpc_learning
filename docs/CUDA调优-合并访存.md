# CUDA调优-合并访存

一个thread负责一个元素相加
```cuda
__global__ void vector_add_kernel_v0(float* a, float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

一个thread负责4个元素相加
```cuda
__global__ void vector_add_kernel_v1(float* a, float* b, float* c, int n) {
    
    // 计算当前线程处理的元素索引
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    // 确保线程索引不超过数组边界
    if(idx < n){
        float4 tempA = FLOAT4(a[idx]);
        float4 tempB = FLOAT4(b[idx]);
        float4 tempC;
        tempC.x = tempA.x + tempB.x;
        tempC.y = tempA.y + tempB.y;
        tempC.z = tempA.z + tempB.z;
        tempC.w = tempA.w + tempB.w;
        FLOAT4(c[idx]) = tempC;
    }
}

```

# 性能结果

```
内核性能测试结果：
数组大小        耗时(ms)        v_0带宽(GB/s)   v1_带宽(GB/s)
-----------------------------------------------------------
     1024          0.004            2.89            2.51
    16384          0.003           56.89           49.55
  1048576          0.018          708.50          708.50
 12582912          0.182          828.11          761.19
134217728          1.883          855.28          770.54
```

- TODO:结果很奇怪，使用FLOAT4后，带宽反而下降了。解决了：是grid没有/4

