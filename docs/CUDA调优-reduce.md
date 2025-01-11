# Reduce操作
Reduce（归约）将一个数组的所有元素通过某种运算（如求和）归约为一个值。本文将介绍CUDA中reduce操作的几种优化方法。

知其然，知其所以然篇

# 基准版本 (Baseline)

![image.png](https://s2.loli.net/2025/01/10/knqDPrK4H3z69hU.png)

如上图，baseline版本中，每个thread先从global memory中读取一个元素，然后通过shared memory将结果传递给下一个thread，直到所有元素相加完毕。图中每一个方格同时对应一个thread和一个数据，红色的格子表示执行加法的线程，而白色的格子表示未执行加法的线程。
代码如下：
其中step不断翻倍，直到step大于blockDim.x，然后每个thread负责一个元素的相加。

```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_baseline(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0) {
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}
```
使用ncu进行测试后， 结果如下：

| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23 | 基准版本，存在warp divergence问题 |
存在的问题：
- GPU中以Warp作为调度单位，每个Warp中包含32个thread，而每个thread负责一个元素的相加，导致Warp中大部分thread处于空闲状态（也就是图中的白色格子线程什么也没做）， 这就是warp divergence问题。

# 优化一：Warp Divergence
在刚刚的例子中，由于每个线程负责相加相邻的元素，导致Warp中大部分thread处于空闲状态，优化的方法就是让工作的thread尽量都在一个warp中。所以在这个版本中，我们将thread于数据的对应给隔离开，具体见下图
![WeChat9ee2a4d81b3fa4fda634ed79a899648c.jpg](https://s2.loli.net/2025/01/10/vyHUibOflpXag7R.jpg)

数据的计算方式保持不变， 但是给线程分配的情况变了，不再是数据下标和线程的下标一一对应，将线程尽可能的聚拢起来，减少warp divegence。

```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v1(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if(threadIdx.x * 2 *s < blockDim.x) {
            int index = threadIdx.x * 2 * s;
            if(index < blockDim.x) {
                sdata[index] += sdata[index + s];
            }
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}
```
解释一下代码为什么这么写：
- step的模式保持不变， 所以for循环没有改变
- 改变的只有每个thread负责的元素下标， 所以if语句中判断条件变了，由于每个thread负责两个元素相加, 为了减少baseline中的白色空槽，所以这里是threadIdx.x * 2, 乘以s是因为每一轮过后， 跳跃的step都翻倍了
## 测试
使用ncu进行测试后， 结果如下：

| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |

思考：这里我是按照一些博客，以及NVIDIA的ppt上来优化的，很显然这种优化思路并不常规，这里就出现了性能下降。
## 存在的问题：
- 每个thread负责两个元素相加， 但是每个thread负责的元素在shared memory中是连续的， 所以会导致bank conflict， 影响性能
从图中可以看到，在第一轮step=1的时,
```
thread_0 : 0, 1
thread_1 : 2, 3
...
thread_15 : 30, 31
thread_16 : 32, 33
...
thread_31 : 62, 63
```
其中thread_0和thread_16负责的元素在shared memory中是在一个bank中的， 所以该warp一次访存需要两次memory transaction。

留一个小问题给读者思考，在baseline中是否存在bank conflict？

# 优化二：bank conflct
在优化一中，每个thread负责两个元素相加，在访问shared memory时， 会导致bank conflict。为了解决这个问题，我们应该研究如何优化访存行为，因此在这里我们应该修改每个thread负责的元素，保证我们一个warp中每个thread负责的元素访存不存在bank conflict。

为了解决bank conflict， 需要让同一个warp中thread尽量每次访存都在不同的bank中，在优化一的版本中，是0和16负责的元素在同一个bank中，所以，只需要将我们每次相加负责的数组元素隔离开就可以了。如下图。
![WeChat7c0ca3a56c05a95369564000ab7f914e.jpg](https://s2.loli.net/2025/01/10/IcUpxShAquZ1Jm8.jpg)

可以看到，每个县城负责的不再是相邻的元素，而是把整个数组切成两个大的快，每次加上相隔blockDim.x/2下标的元素即可， 并且由于这个变化，我们还把threadIdx.x与数组下标对应上了，于是代码变得更简单了。
此时单个warp的访存变为
```
thread_0 : 0, 32
thread_1 : 1, 33
...
thread_15 : 15, 47
thread_16 : 16, 48
...
thread_31 : 31, 63
```
可以看到，每个warp第一次访存为0-32, 第二次访存为32-63，已经没有bank conflict了。再次强调bank conflict是发生在同一个warp中，不同的线程对于同一个bank的访问。
```
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v2(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(int i=blockDim.x/2; i > 0; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}
```

## 性能测试


| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |
| 优化二 | 239.72 | 2.25ms | 解决bank conflict问题 |

在解决bank conflict后， 性能得到了很大的提升。

# 优化三：计算访存并行

在之前的版本中，我们粗暴的将数组直接从global memory全部load到shared memory中， 但是实际上，我们可以在load到shared memory的同时， 进行计算,如下图。
![WeChat694b60707b5b6e9effcd24f03367fa57.jpg](https://s2.loli.net/2025/01/10/oJxK2tDLXPOsdk8.jpg)
从图中可以看出，在从global memory中load到shared memory的同时，进行计算。

代码
```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v3(float *input, float* output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2; i > 0; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

```
## 性能测试
| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |
| 优化二 | 239.72 | 2.25ms | 解决bank conflict问题 |
| 优化三 | 463.91 | 1.16ms | 计算访存并行 |

这里的带宽和耗时提升都很大，其他资料都说把idle线程利用起来，但是这个在ncu的性能报告中并没有体现。

# 优化四：warp reduce

上述的计算流程中， 每一个step的结算结束后，我们都需要同步所有的warp，但实际上，当reduce到一个warp内部时， 我们此时已经不需要sync了，因此我们考虑在最后一个warp内部单独进行reduce。

```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v4(float *input, float* output, int n){
    volatile __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2; i > 32; i/=2) {
        if(threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads(); 
    }
    if(tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4]; 
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}
```
注意， 这里还在sdata中添加了一个volatile关键字， 该关键字的作用是告诉编译器，让编译器不要帮我们使用寄存器重用来优化代码，而是每次都重新从shared memory中读取数据。（留给读者思考：为什么需要这个关键字呢？）

## 性能测试

| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |
| 优化二 | 239.72 | 2.25ms | 解决bank conflict问题 |
| 优化三 | 463.91 | 1.16ms | 计算访存并行 |
| 优化四 | 802.78 | 0.67ms | warp reduce |

这个优化的收益应该来自于warp等待的时延降低了
`smsp__average_warp_latency_issue_stalled_dispatch_stall.pct
`的指标降低了， 相比于v3降低了25%，于是warp能够更快的被调度，访存指令能够更快发射，导致带宽提升。
ps：从这里其实不难看出，reduce的性能瓶颈是memory bound而不是compute，但是如果由于数据以来的原因导致warp stall， 为了更快的发射出访存指令，减少同步关系才能够才来更大的收益。

# 优化五 循环完全展开
其实就是把for的循环变成if然后自己写， 减少编译生成的指令，降低icache miss。

```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v5(float *input, float* output, int n){
    
    volatile __shared__ float sdata[BLOCK_SIZE];
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sdata[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();
    
    if(tid < 128){ 
        sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if(tid < 64) {
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    }
    if(tid < 32){

        sdata[tid] += sdata[tid + 32];
        
        sdata[tid] += sdata[tid + 16];

        sdata[tid] += sdata[tid + 8];

        sdata[tid] += sdata[tid + 4]; 

        sdata[tid] += sdata[tid + 2];

        sdata[tid] += sdata[tid + 1];

    }
    if(tid == 0) output[blockIdx.x] = sdata[0];
}
```
# 性能测试
在看性能测试前， 应该对这个优化心里有个数，其实不会有多少提升，因为该算子的瓶颈并不是icache miss，而是访存带宽。

| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |
| 优化二 | 239.72 | 2.25ms | 解决bank conflict问题 |
| 优化三 | 463.91 | 1.16ms | 计算访存并行 |
| 优化四 | 802.78 | 0.67ms | warp reduce |
| 优化五 | 798.22 | 0.672ms | 循环完全展开 |  

性能几乎没有提升，符合预期。

# 优化六： 使用shuffle指令在warp内部reduce
在优化四中，我们对最后一个warp使用了类似的方法，但是仍然从shared memory中读取数据， 实际上，我们可以使用shuffle指令，在warp内部进行reduce。

## __shfl_down_sync
__shfl_down_sync是CUDA提供的一个warp级别的shuffle指令，用于在同一个warp内的线程之间直接交换数据，无需通过shared memory。其基本语法为:
```cpp
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
```
mask: 用于指定参与shuffle操作的线程掩码,通常使用0xffffffff表示所有线程都参与
var: 要传递的变量值
delta: 向下偏移的线程数
width: warp中参与shuffle的线程数,默认为32(一个warp的大小)

工作原理：
每个线程将自己的var值传给当前线程ID - delta的线程， 然后每个线程接收当前线程ID + delta的线程的值。

优势：
- 避免了shared memory的访问,减少了内存延迟
- 不需要使用__syncthreads()同步
- 在warp内数据交换的效率更高

## 代码部分
在了解了__shfl_down_sync函数后，我们来实现reduce的代码。global memory到shared memory的部门不变，当数据load到shared memory后， 我们使用__shfl_down_sync在warp内部进行reduce。注：此处因为BLOCK_SIZE只有256，因此WARP_SIZE为32是足够的，因为进行第一轮warp reduce后，每个warp只剩下一个数据，那么第一轮过后只剩下BLOCK_SIZE/32个数据，因此这里开辟32的共享内存是绰绰有余的。
第一轮reduce， 每个warp负责的元素是相邻的， 因此delta为16， 第二轮reduce， 每个warp负责的元素是相隔16的， 因此delta为8， 以此类推。
当delta为1时，实际上就是thread_0和thread_1进行相加， 此时每个warp的reduce就完成了。还需要将一个block内的所有warp中的第一个元素通过shared memory来通信， 然后进行最后的reduce。
```cpp
template<int BLOCK_SIZE>
__global__ void reduce_kernel_v6(float *input, float* output, int n){
    const int WARP_SIZE = 32;
    float sum = 0.f;
    float *input_begin = input + blockDim.x * blockIdx.x * 2;
    int tid = threadIdx.x;
    sum = input_begin[tid] + input_begin[tid + blockDim.x];
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    __shared__ float warpLevelSums[32];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;
    if(laneId==0)warpLevelSums[warpId] = sum;
    __syncthreads();
    if(tid < 32){
        sum = (tid < blockDim.x /32) ? warpLevelSums[tid] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    if(tid == 0) output[blockIdx.x] = sum;
}
```

## 性能测试

| 版本 | 带宽 (GB/s) | 耗时 (ms) | 说明 |
|-----|------------|-----------|------|
| Baseline | 166.72 | 3.23ms | 基准版本，存在warp divergence问题 |
| 优化一 | 232.72 | 2.32ms | 解决warp divergence问题，但存在bank conflict |
| 优化二 | 239.72 | 2.25ms | 解决bank conflict问题 |
| 优化三 | 463.91 | 1.16ms | 计算访存并行 |
| 优化四 | 802.78 | 0.67ms | warp reduce |
| 优化五 | 798.22 | 0.672ms | 循环完全展开 |  
| 优化六 | 863.39 | 0.619ms | shuffle reduce |

性能提升很大, 再次思考性能收益来自于哪里，没错，还是warp stalled，在优化五种，还有很多次的shared memory的访存， 而优化六中， 使用shuffle指令， 只需要一次shared_memory的访存，且只有一个同步指令。
