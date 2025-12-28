# Quick Start Guide: Dell 7920 + RTX 4060

This guide gets you running in 10 minutes.

## Prerequisites

Your RTX 4060 needs:
- **NVIDIA Driver**: 535 or newer
- **CUDA Toolkit**: 12.x
- **OS**: Ubuntu 22.04/24.04 or Windows 11 WSL2

---

## Step 1: Verify GPU (2 min)

```bash
# Check driver
nvidia-smi

# Should show:
# NVIDIA GeForce RTX 4060
# Driver Version: 535.xx or higher
```

If not installed:
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

---

## Step 2: Install CUDA Toolkit (5 min)

```bash
# Ubuntu - quick method
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
# Should show: Cuda compilation tools, release 12.x
```

---

## Step 3: Build & Run (3 min)

```bash
# Extract project
tar -xzvf gpu-flight-dynamics.tar.gz
cd gpu-flight-dynamics

# Build CUDA library (sm_89 = RTX 4060)
cd cuda
make CUDA_ARCH=sm_89
make test   # Run tests
make bench  # Run benchmarks
cd ..

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install numpy cupy-cuda12x torch --index-url https://download.pytorch.org/whl/cu121

# Run Python demo
cd python
python flight_dynamics.py
```

---

## Expected Benchmark Results (RTX 4060)

Based on your hardware, you should see approximately:

| Instances | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|--------:|
| 100       | ~50 ms   | ~1 ms    | ~50x    |
| 1,000     | ~500 ms  | ~4 ms    | ~125x   |
| 10,000    | ~5000 ms | ~30 ms   | ~165x   |

The RTX 4060 should achieve **100-200x speedup** at scale!

---

## Troubleshooting

### "nvcc not found"
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Add to ~/.bashrc to make permanent
```

### "CUDA out of memory"
Your RTX 4060 has 8GB. For 10,000 instances:
- States: 10,000 × 12 × 4 bytes = 480 KB
- Controls: 10,000 × 4 × 4 bytes = 160 KB
- Total: < 1 MB (plenty of headroom!)

You can run 100,000+ instances on your GPU.

### "libcudart.so not found"
```bash
sudo ldconfig
# Or add to LD_LIBRARY_PATH
```

---

## Next Steps

1. **Run benchmarks** and save the output for your portfolio
2. **Take screenshots** of nvidia-smi during benchmark
3. **Push to GitHub** with your benchmark results
4. **Write a blog post** about GPU acceleration

---

## GPU Architecture Reference

Your RTX 4060:
- **Architecture**: Ada Lovelace (sm_89)
- **CUDA Cores**: 3072
- **Memory**: 8 GB GDDR6
- **Memory Bandwidth**: 272 GB/s
- **FP32 Performance**: 15.1 TFLOPS

This is more than enough for flight simulation and RL training!
