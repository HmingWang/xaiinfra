#include "tensor.hpp"
#include <iostream>
using namespace std;

int main() {
  cout << "Hello, xAIInfra!" << endl;

  // 创建一个 1024x1024 的矩阵
  Tensor<float> t({1024, 1024});
  std::cout << "Tensor allocated successfully. Size: " << t.total_size
            << std::endl;

  // 测试索引
  t.at({0, 1}) = 3.14f;
  std::cout << "Value at (0, 1): " << t.at({0, 1}) << std::endl;
  return 0;
}