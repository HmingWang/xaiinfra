#pragma once

#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

template <typename T> class Tensor {
public:
  std::vector<size_t> shape; // Shape of the tensor
  std::shared_ptr<T[]> data; // Data of the tensor
  // 对于坐标 $(i, j)$，内存偏移量 $Offset = i \times Stride[0] + j \times
  // Stride[1]$。
  std::vector<size_t> strides; // 新增：存储每一维的步长
  size_t total_size;

  Tensor(std::vector<size_t> s) : shape(s) {
    total_size = 1;
    for (auto d : shape)
      total_size *= d;

    // 初始化 Strides (默认行优先)
    strides.resize(shape.size());
    size_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = current_stride;
      current_stride *= shape[i];
    }

    // 1. 计算对齐后的总字节数
    size_t alignment = 64;
    size_t byte_size = total_size * sizeof(T);
    // 向上取整到 alignment 的倍数
    size_t padded_size = (byte_size + alignment - 1) & ~(alignment - 1);
    // 使用对齐内存分配，这对于后续使用 AVX/SIMD 指令集至关重要
    // 对齐到 64 字节（Cache Line 长度）
    T *raw_ptr = static_cast<T *>(std::aligned_alloc(64, padded_size));
    if (!raw_ptr)
      throw std::runtime_error("Allocation failed");

    data.reset(raw_ptr, [](T *p) { std::free(p); });
    std::fill(data.get(), data.get() + total_size, 0.0f);
  }

  // 真正的索引函数：支持任意维度
  T &at(const std::vector<size_t> &indices) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides[i];
    }
    return data[offset];
  }

  void fill_random(float mean = 0.0f, float stddev = 0.02f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, stddev);
    for (size_t i = 0; i < total_size; ++i) {
      data[i] = d(gen);
    }
  }
};

template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
    // 1. 维度校验
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::runtime_error("Only 2D matrix multiplication is supported.");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("Dimension mismatch: A.cols must equal B.rows.");
    }

    const size_t M = A.shape[0];
    const size_t K = A.shape[1];
    const size_t N = B.shape[1];

    // 2. 创建结果 Tensor (内部会自动分配对齐内存)
    Tensor<T> C({M, N});

    const T* a_ptr = A.data.get();
    const T* b_ptr = B.data.get();
    T* c_ptr = C.data.get();

    // 3. 高性能计算内核 (保持之前的 Tiling + OpenMP 优化)
    const int tile_size = 32;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i += tile_size) {
        for (size_t j = 0; j < N; j += tile_size) {
            for (size_t k = 0; k < K; k += tile_size) {
                
                // 计算当前 Tile 的边界，防止越界
                size_t i_end = std::min(i + tile_size, M);
                size_t j_end = std::min(j + tile_size, N);
                size_t k_end = std::min(k + tile_size, K);

                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        T a_val = a_ptr[ii * K + kk];
                        for (size_t jj = j; jj < j_end; ++jj) {
                            c_ptr[ii * N + jj] += a_val * b_ptr[kk * N + jj];
                        }
                    }
                }
            }
        }
    }

    return C; // 触发 RVO 或移动语义，不会发生大规模内存拷贝
}