import torch
import time

def Test_GPU2CPU_Bandwidth():
    # 测试参数
    size_in_mb = 100  # 数据大小（MB）
    num_trials = 10  # 测试次数
    data_size = size_in_mb * 1024 * 1024  # 转换为字节

    # 创建 GPU 张量
    tensor_gpu = torch.randn(data_size // 4, dtype=torch.float32, device="cuda")

    # 测试 GPU → CPU 传输时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    for _ in range(num_trials):
        tensor_cpu = tensor_gpu.to("cpu")  # 传输数据

    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end) / num_trials  # 平均时间（毫秒）
    bandwidth = (size_in_mb / (elapsed_time / 1000))  # 转换为 MB/s

    print(f"GPU → CPU 带宽: {bandwidth:.2f} MB/s")

def Test_CPU2GPU_Bandwidth():
    # 测试参数
    size_in_mb = 100  # 数据大小（MB）
    num_trials = 10  # 测试次数
    data_size = size_in_mb * 1024 * 1024  # 转换为字节

    # 创建 CPU 张量
    tensor_cpu = torch.randn(data_size // 4, dtype=torch.float32, device="cpu")

    # 测试 CPU → GPU 传输时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    for _ in range(num_trials):
        tensor_gpu = tensor_cpu.to("cuda")  # 传输数据

    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end) / num_trials  # 平均时间（毫秒）
    bandwidth = (size_in_mb / (elapsed_time / 1000))  # 转换为 MB/s

    print(f"CPU → GPU 带宽: {bandwidth:.2f} MB/s")

if __name__ == "__main__":
    Test_GPU2CPU_Bandwidth()
    Test_CPU2GPU_Bandwidth()
