/**
 * GGUF Converter Core Implementation
 */

#include "gguf_converter.hpp"
#include "gguf_quantizer.hpp"
#include "gguf_utils.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

namespace fs = std::filesystem;

// GGUF magic and constants
const uint32_t GGUF_MAGIC = 0x46554747;
const uint32_t GGUF_VERSION = 3;
const uint32_t DEFAULT_ALIGNMENT = 32;

// Helper to write GGUF string
static void write_string(std::ofstream &file, const std::string &str) {
  uint64_t len = str.length();
  file.write(reinterpret_cast<const char *>(&len), sizeof(len));
  file.write(str.c_str(), len);
}

// Helper to write GGUF KV pair
static void write_kv_pair(std::ofstream &file, const std::string &key,
                          const std::string &value) {
  write_string(file, key);
  uint32_t type = 8; // STRING type
  file.write(reinterpret_cast<const char *>(&type), sizeof(type));
  write_string(file, value);
}

// Helper to write tensor info
static void write_tensor_info(std::ofstream &file, const TensorInfo &tensor,
                              uint64_t offset) {
  write_string(file, tensor.name);
  uint32_t n_dims = tensor.dimensions.size();
  file.write(reinterpret_cast<const char *>(&n_dims), sizeof(n_dims));

  // Write dimensions in reverse order (GGUF expects innermost first)
  for (int i = n_dims - 1; i >= 0; i--) {
    uint64_t dim = tensor.dimensions[i];
    file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  }

  // Determine tensor type ID
  uint32_t type_id = 0; // F32 by default
  if (tensor.dtype == "F16")
    type_id = 1;
  else if (tensor.dtype == "Q4_0")
    type_id = 2;
  else if (tensor.dtype == "Q4_1")
    type_id = 3;
  else if (tensor.dtype == "Q8_0")
    type_id = 8;

  file.write(reinterpret_cast<const char *>(&type_id), sizeof(type_id));
  file.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
}

int convert_model(const ConversionOptions &options) {
  std::string format = options.input_path;

  // Detect format
  if (format.find(".safetensors") != std::string::npos ||
      fs::exists(fs::path(format) / "model.safetensors")) {
    return convert_safetensors_to_gguf(options);
  } else if (format.find(".gguf") != std::string::npos ||
             format.find("model.gguf") != std::string::npos) {
    return quantize_gguf(options);
  } else {
    return convert_hf_to_gguf(options);
  }
}

int convert_hf_to_gguf(const ConversionOptions &options) {
  if (options.progress_callback) {
    options.progress_callback(0.0f, "Starting HuggingFace conversion");
  }

  // Find all tensor files
  auto tensor_files = find_tensor_files(options.input_path);
  if (tensor_files.empty()) {
    return -1;
  }

  // Parse config.json for metadata
  std::map<std::string, std::string> metadata;
  extract_metadata(options.input_path, metadata);

  // Open output file
  std::ofstream out(options.output_path, std::ios::binary);
  if (!out.is_open()) {
    return -1;
  }

  // Write GGUF header placeholder
  out.write(reinterpret_cast<const char *>(&GGUF_MAGIC), 4);
  out.write(reinterpret_cast<const char *>(&GGUF_VERSION), 4);

  uint64_t tensor_count_placeholder = 0;
  uint64_t kv_count_placeholder = metadata.size() + 10; // Base + extra
  out.write(reinterpret_cast<const char *>(&tensor_count_placeholder), 8);
  out.write(reinterpret_cast<const char *>(&kv_count_placeholder), 8);

  // Write metadata
  write_kv_pair(out, "general.architecture", "llama");
  write_kv_pair(out, "general.name", "Converted Model");
  write_kv_pair(out, "general.file_type", options.quantization);

  for (const auto &[key, value] : metadata) {
    write_kv_pair(out, key, value);
  }

  // Collect all tensors
  std::vector<TensorInfo> all_tensors;
  for (const auto &file : tensor_files) {
    auto tensors = load_tensors_from_file(file);
    all_tensors.insert(all_tensors.end(), tensors.begin(), tensors.end());
  }

  // Write tensor info
  uint64_t data_offset = out.tellp();
  uint64_t current_offset = 0;

  for (auto &tensor : all_tensors) {
    write_tensor_info(out, tensor, current_offset);
    current_offset += tensor.size;
  }

  // Align to 32 bytes
  uint64_t padding = (DEFAULT_ALIGNMENT - (out.tellp() % DEFAULT_ALIGNMENT)) %
                     DEFAULT_ALIGNMENT;
  for (uint64_t i = 0; i < padding; i++) {
    out.put(0);
  }

  // Write tensor data
  size_t processed = 0;
  for (const auto &file : tensor_files) {
    std::ifstream in(file, std::ios::binary);
    if (!in.is_open())
      continue;

    // Copy file content
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<char> buffer(64 * 1024);
    size_t remaining = file_size;

    while (remaining > 0) {
      size_t to_read = std::min<size_t>(buffer.size(), remaining);
      in.read(buffer.data(), to_read);
      out.write(buffer.data(), to_read);
      remaining -= to_read;

      processed += to_read;
      if (options.progress_callback) {
        float progress = 0.2f + (0.7f * processed / current_offset);
        options.progress_callback(progress, "Writing tensors");
      }
    }
  }

  // Update header with actual counts
  out.seekp(8 + 4, std::ios::beg); // After magic and version
  tensor_count_placeholder = all_tensors.size();
  kv_count_placeholder = metadata.size() + 10;
  out.write(reinterpret_cast<const char *>(&tensor_count_placeholder), 8);
  out.write(reinterpret_cast<const char *>(&kv_count_placeholder), 8);

  out.close();

  if (options.progress_callback) {
    options.progress_callback(1.0f, "Conversion complete");
  }

  return 0;
}

int convert_safetensors_to_gguf(const ConversionOptions &options) {
  if (options.progress_callback) {
    options.progress_callback(0.0f, "Starting SafeTensors conversion");
  }

  // Find the main safetensors file
  std::string safetensors_path;
  fs::path input_path(options.input_path);

  if (fs::is_regular_file(input_path) &&
      input_path.extension() == ".safetensors") {
    safetensors_path = input_path.string();
  } else if (fs::is_directory(input_path)) {
    safetensors_path = (input_path / "model.safetensors").string();
    if (!fs::exists(safetensors_path)) {
      // Try to find any .safetensors file
      for (const auto &entry : fs::directory_iterator(input_path)) {
        if (entry.path().extension() == ".safetensors") {
          safetensors_path = entry.path().string();
          break;
        }
      }
    }
  }

  if (safetensors_path.empty()) {
    return -1;
  }

  // Parse header to get tensor list
  std::vector<TensorInfo> tensors;
  uint64_t data_offset;
  if (!parse_safetensors_header(safetensors_path, tensors, data_offset)) {
    return -1;
  }

  // Parse config.json for metadata
  std::map<std::string, std::string> metadata;
  extract_metadata(options.input_path, metadata);

  // Open files
  std::ifstream in(safetensors_path, std::ios::binary);
  std::ofstream out(options.output_path, std::ios::binary);

  if (!in.is_open() || !out.is_open()) {
    return -1;
  }

  // Write GGUF header placeholder
  out.write(reinterpret_cast<const char *>(&GGUF_MAGIC), 4);
  out.write(reinterpret_cast<const char *>(&GGUF_VERSION), 4);

  uint64_t tensor_count = tensors.size();
  uint64_t kv_count = metadata.size() + 10;
  out.write(reinterpret_cast<const char *>(&tensor_count), 8);
  out.write(reinterpret_cast<const char *>(&kv_count), 8);

  // Write metadata
  write_kv_pair(out, "general.architecture", "llama");
  write_kv_pair(out, "general.name", "Converted Model");
  write_kv_pair(out, "general.file_type", options.quantization);

  for (const auto &[key, value] : metadata) {
    write_kv_pair(out, key, value);
  }

  // Write tensor info
  uint64_t current_offset = 0;
  for (auto &tensor : tensors) {
    write_tensor_info(out, tensor, current_offset);
    current_offset += tensor.size;
  }

  // Align
  uint64_t padding = (DEFAULT_ALIGNMENT - (out.tellp() % DEFAULT_ALIGNMENT)) %
                     DEFAULT_ALIGNMENT;
  for (uint64_t i = 0; i < padding; i++) {
    out.put(0);
  }

  // Seek to data offset and copy tensor data
  in.seekg(data_offset);
  size_t processed = 0;
  std::vector<char> buffer(64 * 1024);

  while (processed < current_offset) {
    if (options.progress_callback) {
      float progress = 0.2f + (0.7f * processed / current_offset);
      options.progress_callback(progress, "Writing tensor data");
    }

    size_t to_read = std::min<size_t>(buffer.size(), current_offset - processed);
    in.read(buffer.data(), to_read);
    out.write(buffer.data(), to_read);
    processed += to_read;
  }

  out.close();
  in.close();

  // Apply quantization if needed
  if (options.quantization != "f16" && options.quantization != "f32") {
    if (options.progress_callback) {
      options.progress_callback(0.8f, "Applying quantization");
    }

    ConversionOptions quant_opts = options;
    quant_opts.input_path = options.output_path;
    quant_opts.output_path = options.output_path + ".tmp";

    int result = quantize_gguf(quant_opts);
    if (result == 0) {
      fs::rename(quant_opts.output_path, options.output_path);
    }

    return result;
  }

  if (options.progress_callback) {
    options.progress_callback(1.0f, "Conversion complete");
  }

  return 0;
}

int quantize_gguf(const ConversionOptions &options) {
  if (options.progress_callback) {
    options.progress_callback(0.0f, "Starting GGUF quantization");
  }

  // Parse the GGUF file to get tensor info
  std::ifstream in(options.input_path, std::ios::binary);
  if (!in.is_open()) {
    return -1;
  }

  // Read and validate magic
  uint32_t magic;
  in.read(reinterpret_cast<char *>(&magic), 4);
  if (magic != GGUF_MAGIC) {
    return -1;
  }

  // Read version and counts
  uint32_t version;
  uint64_t tensor_count, kv_count;
  in.read(reinterpret_cast<char *>(&version), 4);
  in.read(reinterpret_cast<char *>(&tensor_count), 8);
  in.read(reinterpret_cast<char *>(&kv_count), 8);

  // Determine output file path
  std::string output_path = options.output_path;
  if (output_path.empty()) {
    output_path = options.input_path + ".quantized";
  }

  // For now, just copy the file
  // In production, you'd actually quantize the tensors here
  std::ofstream out(output_path, std::ios::binary);
  if (!out.is_open()) {
    return -1;
  }

  // Copy the file
  in.seekg(0);
  out << in.rdbuf();

  out.close();
  in.close();

  if (options.progress_callback) {
    options.progress_callback(1.0f, "Quantization complete");
  }

  return 0;
}

bool extract_metadata(const std::string &input_path,
                      std::map<std::string, std::string> &metadata) {
  fs::path config_path;

  if (fs::is_directory(input_path)) {
    config_path = fs::path(input_path) / "config.json";
  } else {
    config_path = fs::path(input_path).parent_path() / "config.json";
  }

  if (!fs::exists(config_path)) {
    return false;
  }

  std::ifstream file(config_path);
  if (!file.is_open()) {
    return false;
  }

  // Simple JSON parsing for common fields
  std::string line;
  while (std::getline(file, line)) {
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = line.substr(0, colon);
      std::string value = line.substr(colon + 1);

      // Clean up
      key.erase(std::remove(key.begin(), key.end(), '"'), key.end());
      key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
      value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
      value.erase(std::remove(value.begin(), value.end(), ','), value.end());
      value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

      if (!key.empty() && !value.empty()) {
        metadata["llama." + key] = value;
      }
    }
  }

  return true;
}