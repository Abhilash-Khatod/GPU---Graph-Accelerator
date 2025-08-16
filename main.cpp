#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#include <numeric>
#include <random>

// Graph class using adjacency list
class Graph {
private:
    int num_vertices;
    std::vector<std::vector<int>> adj_list;
    std::vector<double> pagerank;
    std::mutex mtx;

public:
    Graph(int vertices) : num_vertices(vertices), adj_list(vertices), pagerank(vertices, 1.0 / vertices) {}

    void add_edge(int src, int dst) {
        if (src >= 0 && src < num_vertices && dst >= 0 && dst < num_vertices) {
            std::lock_guard<std::mutex> lock(mtx);
            adj_list[src].push_back(dst);
        }
    }

    const std::vector<double>& get_pagerank() const { return pagerank; }

    // CPU-based PageRank (single-threaded for baseline)
    void pagerank_cpu(int iterations, double damping = 0.85) {
        std::vector<double> new_pr(num_vertices, 0.0);
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(new_pr.begin(), new_pr.end(), 0.0);
            for (int v = 0; v < num_vertices; ++v) {
                for (int u : adj_list[v]) {
                    new_pr[u] += pagerank[v] / adj_list[v].size();
                }
            }
            for (int v = 0; v < num_vertices; ++v) {
                new_pr[v] = (1.0 - damping) / num_vertices + damping * new_pr[v];
            }
            pagerank = new_pr;
        }
    }

    // Multithreaded PageRank
    void pagerank_cpu_mt(int iterations, int num_threads, double damping = 0.85) {
        std::vector<std::thread> threads;
        auto worker = [&](int start, int end) {
            std::vector<double> local_pr(num_vertices, 0.0);
            for (int iter = 0; iter < iterations; ++iter) {
                std::fill(local_pr.begin(), local_pr.end(), 0.0);
                for (int v = start; v < end; ++v) {
                    for (int u : adj_list[v]) {
                        local_pr[u] += pagerank[v] / adj_list[v].size();
                    }
                }
                std::lock_guard<std::mutex> lock(mtx);
                for (int v = 0; v < num_vertices; ++v) {
                    pagerank[v] = (1.0 - damping) / num_vertices + damping * local_pr[v];
                }
            }
        };

        int chunk = num_vertices / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk;
            int end = (i == num_threads - 1) ? num_vertices : start + chunk;
            threads.emplace_back(worker, start, end);
        }
        for (auto& t : threads) t.join();
    }

    int get_num_vertices() const { return num_vertices; }
    const std::vector<std::vector<int>>& get_adj_list() const { return adj_list; }
};

// CUDA kernel for PageRank iteration
__global__ void pagerank_kernel(int* d_adj_list, int* d_offsets, int* d_degrees, double* d_pagerank,
                               double* d_new_pr, int num_vertices, double damping) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        double sum = 0.0;
        for (int i = d_offsets[v]; i < d_offsets[v] + d_degrees[v]; ++i) {
            int u = d_adj_list[i];
            sum += d_pagerank[u] / d_degrees[u];
        }
        d_new_pr[v] = (1.0 - damping) / num_vertices + damping * sum;
    }
}

// CUDA-based PageRank
void pagerank_gpu(Graph& g, int iterations, double damping = 0.85) {
    int num_vertices = g.get_num_vertices();
    auto adj_list = g.get_adj_list();
    std::vector<double> pagerank = g.get_pagerank();

    // Flatten adjacency list for CUDA
    std::vector<int> flat_adj_list, offsets(num_vertices), degrees(num_vertices);
    int total_edges = 0;
    for (int v = 0; v < num_vertices; ++v) {
        offsets[v] = total_edges;
        degrees[v] = adj_list[v].size();
        total_edges += degrees[v];
        flat_adj_list.insert(flat_adj_list.end(), adj_list[v].begin(), adj_list[v].end());
    }

    // Allocate device memory
    int *d_adj_list, *d_offsets, *d_degrees;
    double *d_pagerank, *d_new_pr;
    cudaMalloc(&d_adj_list, total_edges * sizeof(int));
    cudaMalloc(&d_offsets, num_vertices * sizeof(int));
    cudaMalloc(&d_degrees, num_vertices * sizeof(int));
    cudaMalloc(&d_pagerank, num_vertices * sizeof(double));
    cudaMalloc(&d_new_pr, num_vertices * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_adj_list, flat_adj_list.data(), total_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees, degrees.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pagerank, pagerank.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA kernel configuration
    int threads_per_block = 256;
    int blocks = (num_vertices + threads_per_block - 1) / threads_per_block;

    // Run PageRank iterations
    for (int iter = 0; iter < iterations; ++iter) {
        pagerank_kernel<<<blocks, threads_per_block>>>(d_adj_list, d_offsets, d_degrees,
                                                      d_pagerank, d_new_pr, num_vertices, damping);
        cudaDeviceSynchronize();
        cudaMemcpy(d_pagerank, d_new_pr, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // Copy results back
    cudaMemcpy(pagerank.data(), d_pagerank, num_vertices * sizeof(double), cudaMemcpyDeviceToHost);

    // Update graph's PageRank
    std::lock_guard<std::mutex> lock(g.mtx); // Assuming mutex is public for simplicity
    g.pagerank = pagerank;

    // Free device memory
    cudaFree(d_adj_list);
    cudaFree(d_offsets);
    cudaFree(d_degrees);
    cudaFree(d_pagerank);
    cudaFree(d_new_pr);
}

// Main function to demonstrate usage
int main() {
    const int VERTICES = 1000;
    Graph g(VERTICES);

    // Generate a random graph
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, VERTICES - 1);
    for (int i = 0; i < VERTICES * 10; ++i) {
        g.add_edge(dis(gen), dis(gen));
    }

    // Run CPU-based PageRank
    g.pagerank_cpu(10);
    std::cout << "CPU PageRank (first 5 vertices):\n";
    auto pr = g.get_pagerank();
    for (int i = 0; i < 5; ++i) {
        std::cout << "Vertex " << i << ": " << pr[i] << "\n";
    }

    // Run multithreaded PageRank
    g.pagerank_cpu_mt(10, 4);
    std::cout << "\nMultithreaded PageRank (first 5 vertices):\n";
    pr = g.get_pagerank();
    for (int i = 0; i < 5; ++i) {
        std::cout << "Vertex " << i << ": " << pr[i] << "\n";
    }

    // Run GPU-based PageRank
    pagerank_gpu(g, 10);
    std::cout << "\nGPU PageRank (first 5 vertices):\n";
    pr = g.get_pagerank();
    for (int i = 0; i < 5; ++i) {
        std::cout << "Vertex " << i << ": " << pr[i] << "\n";
    }

    return 0;
}
