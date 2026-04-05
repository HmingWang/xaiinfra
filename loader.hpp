#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "tensor.hpp"

struct TensorInfo {
    std::vector<size_t> shape;
    size_t begin;
    size_t end;
};

#include <vector>
#include <string>
#include <memory>

class ModelLoader {
public:
    void* mapped_data;
    size_t file_size;

    ModelLoader(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        file_size = st.st_size;
        
        // 使用 mmap 映射文件
        mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }

    /**
     * @brief 由 Loader 驱动创建 Tensor
     * @param shape 期望的维度
     * @param offset_in_bytes 该权重在文件中的起始位置
     */
    template<typename T>
    Tensor<T> get_tensor(const std::vector<size_t>& shape, size_t offset_in_bytes) {
        // 1. 安全检查：确保请求的内存范围没有越界
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        size_t required_bytes = total_elements * sizeof(T);

        if (offset_in_bytes + required_bytes > file_size) {
            throw std::runtime_error("Loader Error: Access out of file bounds.");
        }

        // 2. 获取源数据指针
        T* src_ptr = reinterpret_cast<T*>(static_cast<char*>(mapped_data) + offset_in_bytes);

        // 3. 创建 Tensor 对象
        // 注意：这里我们使用之前定义的构造函数，它会申请自己的对齐内存
        Tensor<T> t(shape);

        // 4. 将 mmap 中的数据拷贝到 Tensor 的对齐内存中
        // 这样可以确保 Tensor 内部的数据是 64 字节对齐的，方便后续 SIMD 优化
        std::copy(src_ptr, src_ptr + total_elements, t.data.get());

        // 5. 返回 Tensor 对象 (利用 RVO/移动语义)
        return t;
    }
};


class SafetensorsLoader : public ModelLoader {
private:
    std::map<std::string, TensorInfo> metadata;
    size_t data_section_start;

public:
    SafetensorsLoader(const std::string& path) : ModelLoader(path) {
        // 1. 读取前 8 字节 (Header 长度)
        uint64_t header_size = *reinterpret_cast<uint64_t*>(mapped_data);
        data_section_start = 8 + header_size;

        // 2. 提取 JSON 字符串
        std::string json_str(static_cast<char*>(mapped_data) + 8, header_size);

        // 3. 简单的 JSON 解析 (这里假设你实现了一个 parse_json 函数)
        // 目标是填充 metadata 映射表
        // metadata["model.layers.0.attention.wq.weight"] = { {4096, 4096}, 0, 16777216 };
        parse_metadata(json_str);
    }

    // 真正的自动化接口：只需要名字
    template<typename T>
    Tensor<T> load(const std::string& tensor_name) {
        if (metadata.find(tensor_name) == metadata.end()) {
            throw std::runtime_error("Tensor not found: " + tensor_name);
        }

        const auto& info = metadata[tensor_name];
        // 计算在整个文件中的绝对偏移量
        size_t absolute_offset = data_section_start + info.begin;

        // 调用我们之前的 get_tensor 逻辑
        return get_tensor<T>(info.shape, absolute_offset);
    }

private:
    void parse_metadata(const std::string& json) {
        // 在实际开发中，这里你会用到类似 nlohmann/json 
        // 或者简单的字符串查找逻辑来提取 "shape" 和 "data_offsets"
    }
};