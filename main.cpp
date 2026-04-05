#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <cstdlib>
#include <omp.h>

// ================= 1. 对齐张量与 CPU 优化算子 =================
struct Tensor {
    std::vector<int64_t> shape;
    std::shared_ptr<float> data;
    size_t numel() const {
        size_t n = 1; for(auto d : shape) n *= d; return n;
    }
    float* ptr() { return data.get(); }
    const float* ptr() const { return data.get(); }
};

// 64字节对齐分配（兼容 AVX-512 缓存行）
Tensor make_tensor(std::vector<int64_t> s) {
    size_t n = 1; for(auto d : s) n *= d;
    void* raw;
    posix_memalign(&raw, 64, n * sizeof(float));
    std::fill((float*)raw, (float*)raw + n, 0.0f);
    Tensor t; t.shape = s;
    t.data = std::shared_ptr<float>((float*)raw, [](float* p){ std::free(p); });
    return t;
}

namespace cpu_ops {
    // 缓存友好的 GEMM (MxK @ KxN -> MxN)
    void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=0; i<M; ++i)
            for(int j=0; j<N; ++j) {
                float sum = 0.0f;
                for(int k=0; k<K; ++k) sum += A[i*K + k] * B[k*N + j];
                C[i*N + j] = sum;
            }
    }

    void layer_norm(const float* X, const float* W, const float* B, float* Y, int N, int D, float eps=1e-5) {
        #pragma omp parallel for
        for(int i=0; i<N; ++i) {
            float mean=0, var=0;
            for(int j=0; j<D; ++j) mean += X[i*D+j];
            mean /= D;
            for(int j=0; j<D; ++j) var += (X[i*D+j]-mean)*(X[i*D+j]-mean);
            float inv_std = 1.0f / std::sqrt(var/D + eps);
            for(int j=0; j<D; ++j)
                Y[i*D+j] = W[j] * (X[i*D+j]-mean) * inv_std + B[j];
        }
    }

    void gelu(const float* X, float* Y, int N) {
        const float c = 0.7978845608028654f;
        #pragma omp parallel for
        for(int i=0; i<N; ++i) {
            float v = X[i];
            Y[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
        }
    }

    void softmax(float* logits, int N) {
        float maxv = -1e9f;
        for(int i=0; i<N; ++i) if(logits[i] > maxv) maxv = logits[i];
        float sum = 0.0f;
        for(int i=0; i<N; ++i) {
            float v = std::exp(logits[i] - maxv);
            logits[i] = v; sum += v;
        }
        for(int i=0; i<N; ++i) logits[i] /= sum;
    }
}

// ================= 2. KV Cache 内存池（工业级布局） =================
struct KVCache {
    int layers, hidden, max_seq;
    // 连续内存布局：[layers * max_seq * hidden]
    std::vector<float> k_cache, v_cache;
    int current_seq = 0;

    KVCache(int l, int h, int m) : layers(l), hidden(h), max_seq(m) {
        k_cache.assign(layers * max_seq * hidden, 0.0f);
        v_cache.assign(layers * max_seq * hidden, 0.0f);
    }

    // 获取指定层、指定位置的 K/V 起始指针
    float* k_ptr(int layer, int pos) { return k_cache.data() + (layer * max_seq + pos) * hidden; }
    float* v_ptr(int layer, int pos) { return v_cache.data() + (layer * max_seq + pos) * hidden; }

    // 获取指定层、前 seq_len 个 token 的 K/V 视图起始指针
    float* k_layer_view(int layer, int seq_len) { return k_cache.data() + layer * max_seq * hidden; }
    float* v_layer_view(int layer, int seq_len) { return v_cache.data() + layer * max_seq * hidden; }

    void reset() { current_seq = 0; }
};

// ================= 3. 推理引擎（加载+生成+采样） =================
class InferenceEngine {
public:
    int vocab, hidden, layers, heads;
    // 权重张量
    Tensor embed_w, ln_final_w, ln_final_b, lm_head;
    std::vector<Tensor> ln1_w, ln1_b, ln2_w, ln2_b, Wq, Wk, Wv, Wo, W1, W2;
    
    KVCache kv;
    std::mt19937 rng;

    InferenceEngine(const std::string& bin_path, int max_seq_len)
        : vocab(64), hidden(128), layers(2), heads(4), kv(layers, hidden, max_seq_len) 
    {
        // 初始化随机数生成器
        rng.seed(42);
        load_weights(bin_path);
    }

    void load_weights(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if(!in) throw std::runtime_error("Cannot open weight file");

        auto read_tensor = [&](Tensor& t) {
            in.read(reinterpret_cast<char*>(t.ptr()), t.numel() * sizeof(float));
        };

        // 分配权重空间
        embed_w = make_tensor({vocab, hidden});
        ln_final_w = make_tensor({hidden}); ln_final_b = make_tensor({hidden});
        lm_head = make_tensor({hidden, vocab});

        for(int i=0; i<layers; ++i) {
            ln1_w.push_back(make_tensor({hidden})); ln1_b.push_back(make_tensor({hidden}));
            ln2_w.push_back(make_tensor({hidden})); ln2_b.push_back(make_tensor({hidden}));
            Wq.push_back(make_tensor({hidden, hidden})); Wk.push_back(make_tensor({hidden, hidden}));
            Wv.push_back(make_tensor({hidden, hidden})); Wo.push_back(make_tensor({hidden, hidden}));
            W1.push_back(make_tensor({hidden, hidden*4})); W2.push_back(make_tensor({hidden*4, hidden}));
        }

        // 严格按第2期保存顺序读取
        read_tensor(embed_w); read_tensor(ln_final_w); read_tensor(ln_final_b); read_tensor(lm_head);
        for(int i=0; i<layers; ++i) {
            read_tensor(ln1_w[i]); read_tensor(ln1_b[i]); read_tensor(ln2_w[i]); read_tensor(ln2_b[i]);
            read_tensor(Wq[i]); read_tensor(Wk[i]); read_tensor(Wv[i]); read_tensor(Wo[i]);
            read_tensor(W1[i]); read_tensor(W2[i]);
        }
        std::cout << "Loaded weights from " << path << "\n";
    }

    // 单步前向传播（带 KV Cache）
    float* forward_step(int token, int seq_idx) {
        static Tensor x({1, hidden}), ln_out({1, hidden}), q({1, hidden}), k({1, hidden}), v({1, hidden});
        static Tensor attn_logits({1, 1024}), attn_out({1, hidden}), res({1, hidden});
        static Tensor ln2_out({1, hidden}), h1({1, hidden*4}), h2({1, hidden});

        // 1. Embedding
        const float* emb_row = embed_w.ptr() + token * hidden;
        std::copy(emb_row, emb_row + hidden, x.ptr());

        for(int l=0; l<layers; ++l) {
            // Attention
            cpu_ops::layer_norm(x.ptr(), ln1_w[l].ptr(), ln1_b[l].ptr(), ln_out.ptr(), 1, hidden);
            cpu_ops::matmul(ln_out.ptr(), Wq[l].ptr(), q.ptr(), 1, hidden, hidden);
            cpu_ops::matmul(ln_out.ptr(), Wk[l].ptr(), k.ptr(), 1, hidden, hidden);
            cpu_ops::matmul(ln_out.ptr(), Wv[l].ptr(), v.ptr(), 1, hidden, hidden);

            // 存入 KV Cache
            std::copy(k.ptr(), k.ptr()+hidden, kv.k_ptr(l, seq_idx));
            std::copy(v.ptr(), v.ptr()+hidden, kv.v_ptr(l, seq_idx));

            // 注意力计算：Q(1,h) @ K_cache(seq+1, h)^T -> (1, seq+1)
            int cur_len = seq_idx + 1;
            float* k_view = kv.k_layer_view(l, cur_len);
            // 手动实现 Q @ K^T (1 x cur_len)
            float* logits = attn_logits.ptr();
            #pragma omp parallel for
            for(int j=0; j<cur_len; ++j) {
                float s = 0.0f;
                for(int d=0; d<hidden; ++d) s += q.ptr()[d] * k_view[j*hidden + d];
                logits[j] = s / std::sqrt(hidden);
            }
            cpu_ops::softmax(logits, cur_len);

            // Attn @ V_cache -> (1, h)
            float* v_view = kv.v_layer_view(l, cur_len);
            std::fill(attn_out.ptr(), attn_out.ptr()+hidden, 0.0f);
            #pragma omp parallel for
            for(int d=0; d<hidden; ++d) {
                float sum = 0.0f;
                for(int j=0; j<cur_len; ++j) sum += logits[j] * v_view[j*hidden + d];
                attn_out.ptr()[d] = sum;
            }

            cpu_ops::matmul(attn_out.ptr(), Wo[l].ptr(), res.ptr(), 1, hidden, hidden);
            // 残差1
            for(int d=0; d<hidden; ++d) x.ptr()[d] += res.ptr()[d];

            // FFN
            cpu_ops::layer_norm(x.ptr(), ln2_w[l].ptr(), ln2_b[l].ptr(), ln2_out.ptr(), 1, hidden);
            cpu_ops::matmul(ln2_out.ptr(), W1[l].ptr(), h1.ptr(), 1, hidden, hidden*4);
            cpu_ops::gelu(h1.ptr(), h1.ptr(), hidden*4);
            cpu_ops::matmul(h1.ptr(), W2[l].ptr(), h2.ptr(), 1, hidden*4, hidden);
            // 残差2
            for(int d=0; d<hidden; ++d) x.ptr()[d] += h2.ptr()[d];
        }

        // Final LN & LM Head
        cpu_ops::layer_norm(x.ptr(), ln_final_w.ptr(), ln_final_b.ptr(), x.ptr(), 1, hidden);
        cpu_ops::matmul(x.ptr(), lm_head.ptr(), x.ptr(), 1, hidden, vocab); // 复用 x 作为 logits 输出
        return x.ptr();
    }

    // Top-P + Temperature 采样
    int sample_token(const float* logits, float temp, float top_p) {
        static std::vector<float> probs(vocab);
        static std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);

        // 1. Temperature & Softmax
        for(int i=0; i<vocab; ++i) probs[i] = logits[i] / std::max(temp, 1e-6f);
        cpu_ops::softmax(probs.data(), vocab);

        // 2. 按概率降序排序
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

        // 3. Top-P 截断
        float cum = 0.0f;
        int cut = vocab;
        for(int i=0; i<vocab; ++i) {
            cum += probs[idx[i]];
            if(cum >= top_p) { cut = i + 1; break; }
        }

        // 4. 截断后重新归一化并采样
        float norm_sum = 0.0f;
        for(int i=0; i<cut; ++i) norm_sum += probs[idx[i]];
        std::uniform_real_distribution<float> dis(0.0f, norm_sum);
        float r = dis(rng);
        for(int i=0; i<cut; ++i) {
            r -= probs[idx[i]];
            if(r <= 0.0f) return idx[i];
        }
        return idx[cut-1];
    }

    // 自回归生成
    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens, 
                              float temperature, float top_p, float top_k) {
        kv.reset();
        std::vector<int> output = prompt;
        
        // 预填 Prompt (不生成，只缓存 KV)
        for(size_t i=0; i<prompt.size(); ++i) {
            forward_step(prompt[i], i);
            output.push_back(prompt[i]); // 实际应记录 token，此处简化
        }
        // 修正：prompt 已包含在 output 中，生成从 prompt.size() 开始
        output.pop_back(); // 移除重复添加的最后一个 prompt token

        std::cout << "Prompt tokens: " << prompt.size() << " | Generating...\n";
        for(int t=0; t<max_tokens; ++t) {
            int pos = output.size();
            int token = output.back();
            float* logits = forward_step(token, pos);
            int next = sample_token(logits, temperature, top_p);
            output.push_back(next);
            
            std::cout << "Step " << t+1 << " | Token: " << next << " | Pos: " << pos << "\n";
            if(next == 0) break; // 假设 0 为 EOS
        }
        return output;
    }
};

// ================= 4. 测试入口 =================
int main() {
    try {
        InferenceEngine engine("model_weights.bin", 512);
        std::vector<int> prompt = {10, 25, 30}; // 示例 prompt
        auto gen = engine.generate(prompt, 20, 0.8f, 0.9f, 0.0f);
        
        std::cout << "\n=== Generated Sequence ===\n";
        for(int t : gen) std::cout << t << " ";
        std::cout << "\nDone.\n";
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}