import torch
import triton
import triton.language as tl
import time
DEVICE = torch.device("cuda:0")



@triton.jit
def vector_add_kernel(
    x_ptr,  # 第一个输入向量的指针
    y_ptr,  # 第二个输入向量的指针
    output_ptr,  # 输出向量的指针
    n_elements,  # 向量的元素数量
    BLOCK_SIZE: tl.constexpr,  # 每个线程块处理的元素数量
):
    # 计算当前线程块的起始位置
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 计算偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 确保不会越界
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 执行加法运算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x, device=DEVICE)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    # 手动计时
    start = time.time()
    if provider == 'torch':
        for _ in range(100):  # 重复多次以获得更稳定的测量
            z = x + y
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
    elif provider == 'triton':
        z = add(x, y)
        torch.cuda.synchronize()
    end = time.time()
    
    # 计算时间
    elapsed_ms = (end - start) * 1000 / 100  # 平均每次操作的时间（毫秒）
    
    # 计算带宽
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(elapsed_ms)

# 测试代码
if __name__ == "__main__":
    sizes = [2**i for i in range(12, 25, 3)]
    for size in sizes:
        triton_gbps = benchmark(size, 'triton')
        # torch_gbps = benchmark(size, 'torch')
        print(f"Size: {size}, Triton: {triton_gbps:.2f} GB/s")


