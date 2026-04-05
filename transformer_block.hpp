#include "tensor.hpp"

template<typename T>
class TransformerBlock {
public:
    // 权重 Tensor
    Tensor<T> wq, wk, wv, wo;          // Attention 投影
    Tensor<T> w1, w2, w3;              // MLP 层 (Llama 使用 SwiGLU，需要三块权重)
    Tensor<T> ln1_gamma, ln1_beta;     // 第一个 LayerNorm
    Tensor<T> ln2_gamma, ln2_beta;     // 第二个 LayerNorm

    // 从 Loader 自动化初始化
    TransformerBlock(SafetensorsLoader& loader, int layer_idx) {
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

    Tensor<T> forward(Tensor<T>& x) {
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
};