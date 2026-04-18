/**
 * GGUF Bridge Utilities
 * Helper functions for file I/O, format detection, and tensor handling
 */

#ifndef GGUF_UTILS_HPP
#define GGUF_UTILS_HPP

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct TensorInfo {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> dimensions;
  uint64_t size;
  uint64_t offset;
};

// Find all tensor files in a model directory
std::vector<std::string> find_tensor_files(const std::string &input_path);

// Count tensors in a file
size_t count_tensors_in_file(const std::string &file_path);

// Count tensors in SafeTensors file
size_t count_tensors_in_safetensors(const std::string &file_path);

// Count tensors in PyTorch file
size_t count_tensors_in_pytorch(const std::string &file_path);

// Load tensors from a file (SafeTensors or PyTorch)
std::vector<TensorInfo> load_tensors_from_file(const std::string &file_path);

// Parse SafeTensors header and load tensors
std::vector<TensorInfo> load_tensors_from_safetensors(const std::string &file_path);

// Parse PyTorch binary format and load tensors
std::vector<TensorInfo> load_tensors_from_pytorch(const std::string &file_path);

// Parse SafeTensors header (legacy)
bool parse_safetensors_header(const std::string &file_path,
                              std::vector<TensorInfo> &tensors,
                              uint64_t &data_offset);

// Parse PyTorch binary format (legacy)
bool parse_pytorch_file(const std::string &file_path,
                        std::vector<TensorInfo> &tensors);

#endif // GGUF_UTILS_HPP