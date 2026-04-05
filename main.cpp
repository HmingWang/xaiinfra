#include "tensor.hpp"
#include <iostream>
using namespace std;

void test_infrastructure_smoke();

int main() {
  cout << "Hello, xAIInfra!" << endl;
  test_infrastructure_smoke();
  return 0;
}


void test_infrastructure_smoke() {
    // 1. 初始化基础设施
    // 假设 Hidden_Dim = 2 (简单起见)
    size_t hidden_dim = 2;
    
    // 模拟输入 X: [1, 0.5] (代表某个词的原始特征)
    Tensor<float> x({1, hidden_dim});
    x.data.get()[0] = 1.0f; x.data.get()[1] = 0.5f;

    // 模拟权重 W_q: [2, 2] (单位矩阵，保持特征不变)
    Tensor<float> w_q({hidden_dim, hidden_dim});
    w_q.data.get()[0] = 1.0f; w_q.data.get()[1] = 0.0f;
    w_q.data.get()[2] = 0.0f; w_q.data.get()[3] = 1.0f;

    // 模拟两个 Key: [2, 2]
    // Row 0: [1.0, 0.6] (香蕉，和苹果很像)
    // Row 1: [-1.0, -0.5] (扳手，方向完全相反)
    Tensor<float> keys({2, hidden_dim});
    keys.at({0, 0}) = 1.0f;  keys.at({0, 1}) = 0.6f;
    keys.at({1, 0}) = -1.0f; keys.at({1, 1}) = -0.5f;

    // --- 开始流水线计算 ---

    // Step A: 计算 Query (MatMul)
    // q = x * w_q
    Tensor<float> q = matmul(x, w_q); 
    std::cout << "Step 1: Query calculated." << std::endl;

    // Step B: 计算 Scores (Attention Map)
    // scores = q * K^T
    keys.transpose(); // 变幻布局
    keys.contiguous(); // 确保连续性 (验证你的 contiguous 函数)
    Tensor<float> scores = matmul(q, keys); // 结果应为 [1, 2]

    // Step C: 归一化 (Softmax)
    // 验证你的 Softmax 是否能拉开差距
    softmax(scores);

    // --- 3. 结果验证 ---
    float score_banana = scores.data.get()[0];
    float score_wrench = scores.data.get()[1];

    std::cout << "--- Validation Result ---" << std::endl;
    std::cout << "Attention to 'Banana': " << score_banana << std::endl;
    std::cout << "Attention to 'Wrench': " << score_wrench << std::endl;

    if (score_banana > 0.8f && score_wrench < 0.2f) {
        std::cout << "SUCCESS: Infrastructure is working correctly!" << std::endl;
    } else {
        std::cout << "FAILURE: Math or Memory alignment issue." << std::endl;
    }
}