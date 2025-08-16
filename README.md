# 🚀 GPU-Graph-Accelerator

This project implements **PageRank** on graphs of up to **1000 vertices** using three approaches:
1. **Single-threaded CPU** (baseline)  
2. **Multi-threaded CPU** (with efficient thread-level parallelism)  
3. **GPU acceleration with CUDA**  

The implementation achieves significant performance improvements, reducing PageRank computation time from **~1 sec (CPU)** to **~0.02 sec (GPU)** for 10 iterations on a 1000-vertex graph.

---

## ✨ Features
- PageRank with **0.85 damping factor**  
- Single-threaded CPU baseline implementation  
- Multi-threaded CPU implementation (4 threads)  
- GPU-accelerated implementation using **CUDA kernels**  
- Random graph generator for benchmarking  
- Prints PageRank values for sample vertices  

---

## ⚙️ Build & Run

### Requirements
- **C++11 or higher**  
- **CUDA Toolkit** installed for GPU support  
