/**
 * GGUF Converter Core
 * Main conversion logic using llama.cpp internals
 */

#ifndef GGUF_CONVERTER_HPP
#define GGUF_CONVERTER_HPP

#include <functional>
#include <string>
#include <vector>
#include <map>

struct ConversionOptions {
  std::string input_path;
  std::string output_path;
  std::string model_type;
  std::string quantization;
  int num_threads;
  bool vocab_only;
  bool use_mmap;
  std::vector<std::pair<std::string, std::string>> metadata_overrides;
  std::function<void(float, const std::string &)> progress_callback;
  std::function<void(int, const std::string &)> log_callback;
};

// Main conversion function
int convert_model(const ConversionOptions &options);

// Convert HuggingFace format to GGUF
int convert_hf_to_gguf(const ConversionOptions &options);

// Convert SafeTensors to GGUF
int convert_safetensors_to_gguf(const ConversionOptions &options);

// Quantize existing GGUF file
int quantize_gguf(const ConversionOptions &options);

// Extract model metadata
bool extract_metadata(const std::string &input_path,
                      std::map<std::string, std::string> &metadata);

#endif // GGUF_CONVERTER_HPP