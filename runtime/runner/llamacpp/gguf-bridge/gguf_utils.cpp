/**
 * GGUF Bridge Utilities Implementation
 */

#include "gguf_utils.hpp"
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::vector<std::string> find_tensor_files(const std::string &input_path) {
  std::vector<std::string> files;
  fs::path path(input_path);

  if (fs::is_directory(path)) {
    // Look for SafeTensors files
    for (const auto &entry : fs::directory_iterator(path)) {
      std::string ext = entry.path().extension().string();
      std::string filename = entry.path().filename().string();

      if (ext == ".safetensors") {
        files.push_back(entry.path().string());
      } else if (filename == "pytorch_model.bin" ||
                 filename.find("pytorch_model-") == 0) {
        files.push_back(entry.path().string());
      } else if (filename == "model.safetensors") {
        files.push_back(entry.path().string());
      }
    }

    // Sort to ensure consistent order
    std::sort(files.begin(), files.end());
  } else if (fs::is_regular_file(path)) {
    files.push_back(path.string());
  }

  return files;
}

size_t count_tensors_in_file(const std::string &file_path) {
  std::string ext = fs::path(file_path).extension().string();

  if (ext == ".safetensors") {
    return count_tensors_in_safetensors(file_path);
  } else if (ext == ".bin" || ext == ".pt" || ext == ".pth") {
    return count_tensors_in_pytorch(file_path);
  }

  return 0;
}

size_t count_tensors_in_safetensors(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return 0;

  // Read header length (8-byte little-endian)
  uint64_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));

  if (header_len > 100 * 1024 * 1024) { // Sanity check: max 100MB header
    return 0;
  }

  // Read header JSON
  std::string header_json(header_len, '\0');
  file.read(&header_json[0], header_len);

  try {
    json header = json::parse(header_json);
    size_t count = 0;

    for (auto &[key, value] : header.items()) {
      if (key != "__metadata__") {
        count++;
      }
    }

    return count;
  } catch (const std::exception &e) {
    return 0;
  }
}

size_t count_tensors_in_pytorch(const std::string &file_path) {
  // PyTorch files are pickled Python objects
  // For now, we'll approximate by scanning for tensor markers
  // In production, you'd use libtorch or a proper parser

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return 0;

  // Get file size
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Simple heuristic: count occurrences of tensor marker patterns
  const char *tensor_marker = "torch.Tensor";
  const size_t marker_len = strlen(tensor_marker);

  std::vector<char> buffer(4096);
  size_t count = 0;
  size_t pos = 0;

  while (pos < file_size) {
    size_t to_read = std::min(buffer.size(), file_size - pos);
    file.read(buffer.data(), to_read);

    for (size_t i = 0; i < to_read - marker_len; i++) {
      if (memcmp(buffer.data() + i, tensor_marker, marker_len) == 0) {
        count++;
      }
    }

    pos += to_read;
  }

  return count;
}

std::vector<TensorInfo> load_tensors_from_file(const std::string &file_path) {
  std::string ext = fs::path(file_path).extension().string();

  if (ext == ".safetensors") {
    return load_tensors_from_safetensors(file_path);
  } else if (ext == ".bin" || ext == ".pt" || ext == ".pth") {
    return load_tensors_from_pytorch(file_path);
  }

  return {};
}

std::vector<TensorInfo>
load_tensors_from_safetensors(const std::string &file_path) {
  std::vector<TensorInfo> tensors;

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return tensors;

  // Read header length
  uint64_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));

  if (header_len == 0 || header_len > 100 * 1024 * 1024) {
    return tensors;
  }

  // Read header JSON
  std::string header_json(header_len, '\0');
  file.read(&header_json[0], header_len);

  uint64_t data_offset = 8 + header_len;

  try {
    json header = json::parse(header_json);

    for (auto &[name, info] : header.items()) {
      if (name == "__metadata__")
        continue;

      TensorInfo tensor;
      tensor.name = name;
      tensor.dtype = info["dtype"];

      // Parse shape
      for (auto &dim : info["shape"]) {
        tensor.dimensions.push_back(dim);
      }

      // Calculate size based on dtype
      size_t element_size = 4; // default F32
      if (tensor.dtype == "F32")
        element_size = 4;
      else if (tensor.dtype == "F16" || tensor.dtype == "BF16")
        element_size = 2;
      else if (tensor.dtype == "I8" || tensor.dtype == "U8")
        element_size = 1;
      else if (tensor.dtype == "I16" || tensor.dtype == "U16")
        element_size = 2;
      else if (tensor.dtype == "I32" || tensor.dtype == "U32")
        element_size = 4;
      else if (tensor.dtype == "I64" || tensor.dtype == "U64")
        element_size = 8;

      size_t num_elements = 1;
      for (auto dim : tensor.dimensions)
        num_elements *= dim;
      tensor.size = num_elements * element_size;

      // Get offsets
      if (info.contains("data_offsets")) {
        tensor.offset = info["data_offsets"][0];
      }

      tensors.push_back(tensor);
    }
  } catch (const std::exception &e) {
    // Parse error
  }

  return tensors;
}

std::vector<TensorInfo>
load_tensors_from_pytorch(const std::string &file_path) {
  // PyTorch format is complex; for production, use libtorch
  // This is a simplified placeholder
  std::vector<TensorInfo> tensors;

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return tensors;

  // Get file size
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Placeholder: return a single tensor representing the whole file
  TensorInfo tensor;
  tensor.name = fs::path(file_path).stem().string();
  tensor.dtype = "F32";
  tensor.dimensions.push_back(file_size / 4);
  tensor.size = file_size;
  tensor.offset = 0;
  tensors.push_back(tensor);

  return tensors;
}

bool parse_safetensors_header(const std::string &file_path,
                              std::vector<TensorInfo> &tensors,
                              uint64_t &data_offset) {
  tensors = load_tensors_from_safetensors(file_path);
  if (tensors.empty())
    return false;

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open())
    return false;

  uint64_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  data_offset = 8 + header_len;

  return true;
}

bool parse_pytorch_file(const std::string &file_path,
                        std::vector<TensorInfo> &tensors) {
  tensors = load_tensors_from_pytorch(file_path);
  return !tensors.empty();
}