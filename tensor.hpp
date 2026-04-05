#pragma once

#include <memory>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

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

  bool is_contiguous() const {
    size_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      if (strides[i] != expected_stride)
        return false;
      expected_stride *= shape[i];
    }
    return true;
  }

  void contiguous() {
    if (is_contiguous()) return; // 已经是连续的，直接返回

    // 1. 创建一个新的临时连续存储
    T* new_raw_ptr = static_cast<T*>(std::aligned_alloc(64, ((total_size * sizeof(T) + 63) / 64) * 64));
    
    // 2. 将非连续的数据按逻辑顺序拷贝过去
    // 这里我们可以利用多维迭代器或者简单的多重循环
    // 下面以 2D 为例演示逻辑：
    #pragma omp parallel for
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            new_raw_ptr[i * shape[1] + j] = this->at({i, j});
        }
    }

    // 3. 更新指针和步长
    data.reset(new_raw_ptr, [](T* p) { std::free(p); });
    
    // 重新计算标准步长（行优先）
    size_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
}

  void transpose() {
    if (shape.size() < 2)
      return;

    // 默认交换最后两个维度（这是深度学习中最常见的转置行为）
    size_t d1 = shape.size() - 2;
    size_t d2 = shape.size() - 1;

    std::swap(shape[d1], shape[d2]);
    std::swap(strides[d1], strides[d2]);

    // 注意：内存里的数据一行都没动，只是我们“看”它的顺序变了
  }

  void reshape(std::vector<size_t> new_shape) {
    // 1. 校验总大小是否匹配
    size_t new_total_size = 1;
    for (auto d : new_shape)
      new_total_size *= d;

    if (new_total_size != this->total_size) {
      throw std::runtime_error("Reshape failed: total element count mismatch.");
    }

    // 2. 更新 shape
    this->shape = new_shape;

    // 3. 重新计算 strides (假设依然保持行优先存储)
    this->strides.resize(shape.size());
    size_t current_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = current_stride;
      current_stride *= shape[i];
    }

    // 注意：data 智能指针完全没动，内存原地复用
  }

  Tensor<T> operator*(const Tensor<T> &B) { return matmul(*this, B); }
};

template <typename T> Tensor<T> matmul(const Tensor<T> &A, const Tensor<T> &B) {
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

  const T *a_ptr = A.data.get();
  const T *b_ptr = B.data.get();
  T *c_ptr = C.data.get();

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

#include <cmath>
#include <algorithm>

template<typename T>
void softmax(Tensor<T>& tensor) {
    if (!tensor.is_contiguous()) tensor.contiguous();

    size_t last_dim = tensor.shape.back();
    size_t num_rows = tensor.total_size / last_dim;
    T* data = tensor.data.get();

    #pragma omp parallel for
    for (size_t i = 0; i < num_rows; ++i) {
        T* row = data + i * last_dim;

        // 1. 找到当前行的最大值 (Safe Softmax 第一步)
        T max_val = *std::max_element(row, row + last_dim);

        // 2. 计算指数和
        T sum = 0;
        for (size_t j = 0; j < last_dim; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        // 3. 归一化
        T inv_sum = 1.0 / sum;
        for (size_t j = 0; j < last_dim; ++j) {
            row[j] *= inv_sum;
        }
    }
}

template<typename T>
Tensor<T> self_attention(Tensor<T>& Q, Tensor<T>& K, Tensor<T>& V) {
    // 假设输入形状都是 [Seq_Len, Hidden_Dim]
    size_t seq_len = Q.shape[0];
    size_t d_k = Q.shape[1];

    // 第一步：计算 Q * K^T (Scores)
    // 注意：K 需要转置
    K.transpose(); // 变幻步长，数据不动
    // 由于我们的 matmul 暂时假设输入是连续的，这里必须执行一次同步
    K.contiguous(); 
    
    Tensor<T> scores = matmul(Q, K); // 结果形状 [Seq_Len, Seq_Len]

    // 第二步：缩放 (Scaling)
    T scale = 1.0 / std::sqrt(static_cast<T>(d_k));
    T* scores_ptr = scores.data.get();
    #pragma omp parallel for
    for (size_t i = 0; i < scores.total_size; ++i) {
        scores_ptr[i] *= scale;
    }

    // 第三步：Softmax (按行归一化)
    // 每一行代表当前词对全句所有词的注意力分配
    softmax(scores);

    // 第四步：计算 Attention_Weights * V
    // V 形状 [Seq_Len, Hidden_Dim]
    Tensor<T> output = matmul(scores, V); // 结果形状 [Seq_Len, Hidden_Dim]

    return output;
}

template<typename T>
void layer_norm(
    Tensor<T>& input, 
    const Tensor<T>& gamma, // 缩放参数 [Hidden_Dim]
    const Tensor<T>& beta,  // 平移参数 [Hidden_Dim]
    float eps = 1e-5f
) {
    size_t seq_len = input.shape[0];
    size_t hidden_dim = input.shape[1];
    T* data = input.data.get();
    const T* g_ptr = gamma.data.get();
    const T* b_ptr = beta.data.get();

    // 每一行（每一个 Token）独立进行归一化
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        T* row = data + i * hidden_dim;

        // 1. 计算均值
        T sum = 0;
        for (size_t j = 0; j < hidden_dim; ++j) sum += row[j];
        T mean = sum / hidden_dim;

        // 2. 计算方差
        T var_sum = 0;
        for (size_t j = 0; j < hidden_dim; ++j) {
            T diff = row[j] - mean;
            var_sum += diff * diff;
        }
        T var = var_sum / hidden_dim;

        // 3. 标准化并应用 Gamma/Beta
        T inv_std = 1.0 / std::sqrt(var + eps);
        for (size_t j = 0; j < hidden_dim; ++j) {
            row[j] = (row[j] - mean) * inv_std * g_ptr[j] + b_ptr[j];
        }
    }
}