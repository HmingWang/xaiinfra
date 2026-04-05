#include "tensor.hpp"

template <typename T> class TransformerBlock {
public:
  // 权重 Tensor
  Tensor<T> wq, wk, wv, wo;      // Attention 投影
  Tensor<T> w1, w2, w3;          // MLP 层 (Llama 使用 SwiGLU，需要三块权重)
  Tensor<T> ln1_gamma, ln1_beta; // 第一个 LayerNorm
  Tensor<T> ln2_gamma, ln2_beta; // 第二个 LayerNorm

  // 从 Loader 自动化初始化
  TransformerBlock(SafetensorsLoader &loader, int layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // 自动化加载所有权重
    wq = loader.load<T>(prefix + "self_attn.q_proj.weight");
    wk = loader.load<T>(prefix + "self_attn.k_proj.weight");
    wv = loader.load<T>(prefix + "self_attn.v_prod.weight");
    wo = loader.load<T>(prefix + "self_attn.o_proj.weight");

    w1 = loader.load<T>(prefix + "mlp.gate_proj.weight");
    w2 = loader.load<T>(prefix + "mlp.down_proj.weight");
    w3 = loader.load<T>(prefix + "mlp.up_proj.weight");

    ln1_gamma = loader.load<T>(prefix + "input_layernorm.weight");
    ln2_gamma = loader.load<T>(prefix + "post_attention_layernorm.weight");
    // 注意：Llama 使用 RMSNorm，通常没有 beta（偏置）
  }

  Tensor<T> forward(Tensor<T> &x) {
    // --- 1. Attention 部分 ---
    Tensor<T> norm_1 = x; // 拷贝一份用于归一化
    layer_norm(norm_1, ln1_gamma, ln1_beta);

    // 执行我们之前写的 Attention 逻辑
    Tensor<T> attn_out = multi_head_attention(norm_1, wq, wk, wv, wo);

    // 残差连接 1: x = x + attn_out
    add_inplace(x, attn_out);

    // --- 2. MLP 部分 ---
    Tensor<T> norm_2 = x;
    layer_norm(norm_2, ln2_gamma, ln2_beta);

    // Llama 的 MLP 通常是: down(silu(gate(x)) * up(x))
    Tensor<T> ffn_out = mlp_swiglu(norm_2, w1, w2, w3);

    // 残差连接 2: x = x + ffn_out
    add_inplace(x, ffn_out);

    return x;
  }

  Tensor<T> mlp_swiglu(Tensor<T> &x, const Tensor<T> &w1, const Tensor<T> &w2,
                       const Tensor<T> &w3) {
    // 1. 映射到中间高维空间 (通常是 Hidden_Dim 的 8/3 倍)
    Tensor<T> gate = matmul(x, w1); // w1 是 Gate 投影
    Tensor<T> up = matmul(x, w3);   // w3 是 Up 投影

    // 2. 逐元素执行 SiLU 并与 up 相乘
    T *g_ptr = gate.data;
    T *u_ptr = up.data;
#pragma omp parallel for
    for (size_t i = 0; i < gate.total_size; ++i) {
      // SiLU: g * sigmoid(g)
      T silu = g_ptr[i] * (1.0f / (1.0f + std::exp(-g_ptr[i])));
      g_ptr[i] = silu * u_ptr[i]; // 融合 Gate 和 Up
    }

    // 3. 映射回原始 Hidden_Dim
    return matmul(gate, w2); // w2 是 Down 投影
  }

  void apply_rope(Tensor<T> &t, int start_pos) {
    size_t seq_len = t.shape[1]; // 假设形状为 [Num_Heads, Seq, Head_Dim]
    size_t num_heads = t.shape[0];
    size_t head_dim = t.shape[2];

#pragma omp parallel for collapse(2)
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t s = 0; s < seq_len; ++s) {
        int pos = start_pos + s;
        T *vec = t.data + (h * seq_len * head_dim) + (s * head_dim);

        for (size_t i = 0; i < head_dim / 2; ++i) {
          // 计算旋转频率 theta
          float theta = pos / std::pow(10000.0f, (float)(2 * i) / head_dim);
          float cos_t = std::cos(theta);
          float sin_t = std::sin(theta);

          // 旋转复数平面上的点 [v0, v1]
          T v0 = vec[2 * i];
          T v1 = vec[2 * i + 1];
          vec[2 * i] = v0 * cos_t - v1 * sin_t;
          vec[2 * i + 1] = v0 * sin_t + v1 * cos_t;
        }
      }
    }
  }
  void add_inplace(Tensor<T> &a, const Tensor<T> &b) {
    if (a.total_size != b.total_size)
      throw std::runtime_error("Size mismatch");

    T *a_ptr = a.data;
    const T *b_ptr = b.data;
#pragma omp parallel for
    for (size_t i = 0; i < a.total_size; ++i) {
      a_ptr[i] += b_ptr[i];
    }
  }
};