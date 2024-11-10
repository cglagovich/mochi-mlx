import math
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from contextlib import nullcontext
import tabulate
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

@dataclass
class BenchConfig:
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    
    def __str__(self):
        return f"B={self.batch_size}, S={self.seq_len}, H={self.num_heads}, D={self.head_dim}"

def create_attention_inputs(config: BenchConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates random query, key, value tensors and an attention mask for benchmarking."""
    q = torch.randn(
        (config.batch_size, config.num_heads, config.seq_len, config.head_dim),
        device=config.device,
        dtype=config.dtype,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.seq_len, config.head_dim),
        device=config.device,
        dtype=config.dtype,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.seq_len, config.head_dim),
        device=config.device,
        dtype=config.dtype,
    )
    return q, k, v

def benchmark_backend(
    config: BenchConfig,
    backend: str,
    warmup: int = 5,
    repeats: int = 20,
) -> Dict[str, float]:
    """Benchmarks a specific SDPA backend with given config."""
    q, k, v = create_attention_inputs(config)
    
    backend = {
        "sdpa_FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "sdpa_CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
        "sdpa_EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    }[backend]
    ctx = sdpa_kernel(backend)
    kernel = backend

    # Warmup
    for _ in range(warmup):
        with ctx:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None
            )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    timings = []
    
    with ctx:
        for _ in range(repeats):
            start_event.record()
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None
            )
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    
    avg_time = sum(timings) / len(timings)
    
    # Calculate FLOPs
    # FLOPS = 2 * B * H * S * S * D (QK^T) + 2 * B * H * S * S * D (AV)
    flops = 2 * config.batch_size * config.num_heads * (config.seq_len ** 2) * config.head_dim * 2
    
    return {
        "avg_time_ms": avg_time,
        "flops": flops,
        "tflops_per_sec": (flops / (avg_time / 1000)) / 1e12,
        "kernel": kernel
    }

def run_benchmarks(configs: List[BenchConfig]) -> str:
    """Runs benchmarks for all configs and backends, returns markdown table."""
    backends = ["sdpa_FLASH_ATTENTION", "sdpa_EFFICIENT_ATTENTION", "sdpa_CUDNN_ATTENTION"]
    results = []
    
    for config in configs:
        print(f"\nBenchmarking config: {config}")
        for backend in backends:
            print(f"Testing {backend} backend...")
            try:
                stats = benchmark_backend(config, backend)
                results.append({
                    "Config": str(config),
                    "Backend": backend,
                    "Time (ms)": f"{stats['avg_time_ms']:.2f}",
                    "TFLOPs/sec": f"{stats['tflops_per_sec']:.2f}",
                    "Kernel": stats['kernel']
                })
            except RuntimeError as e:
                print(f"Error with {backend} backend: {e}")
                results.append({
                    "Config": str(config),
                    "Backend": backend,
                    "Time (ms)": "FAILED",
                    "TFLOPs/sec": "FAILED",
                    "Kernel": "N/A"
                })
    
    # Create markdown table
    headers = ["Config", "Backend", "Time (ms)", "TFLOPs/sec", "Kernel"]
    return tabulate.tabulate(
        [{k: row[k] for k in headers} for row in results],
        headers=headers,
        tablefmt="pipe"
    )

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    # Example configurations to test
    multiples_of = [1, 64, 128, 256, 512, 1024]
    base_seq_len = 44557
    configs = [
        BenchConfig(
            batch_size=1,
            seq_len=int(math.ceil(base_seq_len / multiple) * multiple),
            num_heads=24,
            head_dim=128,
            dtype=torch.bfloat16,
            device=torch.device("cuda")
        ) for multiple in multiples_of
    ]
    
    # Run benchmarks and print results
    print("\nBenchmark Results:")
    print(run_benchmarks(configs)) 