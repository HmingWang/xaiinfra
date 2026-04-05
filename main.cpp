#include "tensor.hpp"
#include <iostream>
using namespace std;

int main() {
  cout << "Hello, xAIInfra!" << endl;

  // 创建一个 2x3 的矩阵
  Tensor<float> m({2, 3});
  Tensor<float> n({3, 4}); // 创建一个 4x4 的矩阵
  auto p = m * n;// 创建一个 2x4 的矩阵

  // 测试索引
  p.at({0, 1}) = 3.14f;
  std::cout << "Value at (0, 1): " << p.total_size << std::endl;
  return 0;
}