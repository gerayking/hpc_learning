import torch
import triton
import triton.language as tl

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

# 包装函数，方便调用
def vector_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "输入张量必须在GPU上"
    assert x.shape == y.shape, "输入张量形状必须相同"
    assert x.is_contiguous() and y.is_contiguous(), "输入张量必须是连续的"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 定义每个线程块处理的元素数量
    BLOCK_SIZE = 1024
    
    # 计算需要启动的网格大小
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动kernel
    vector_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    x = torch.randn(1000000, device="cuda")
    y = torch.randn(1000000, device="cuda")
    
    # 使用Triton实现的向量加法
    output_triton = vector_add(x, y)
    
    # 使用PyTorch原生实现作为参考
    output_torch = x + y
    
    # 验证结果
    print("最大误差:", torch.max(torch.abs(output_triton - output_torch)))
