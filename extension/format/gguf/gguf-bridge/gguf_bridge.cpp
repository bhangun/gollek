/**
 * GGUF Bridge Implementation
 *
 * Implements the C API bridge to llama.cpp conversion functionality
 */

#include "gguf_bridge.hpp"
#include <llama.h>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>

namespace fs = std::filesystem;

// Thread-local error storage
static thread_local std::string g_last_error;

// Version string
static const char* GGUF_BRIDGE_VERSION = "1.0.0";

// Available quantization types
static const char* QUANTIZATION_TYPES[] = {
    "f32", "f16",
    "q4_0", "q4_1", "q5_0", "q5_1",
    "q8_0", "q8_1",
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m",
    "q6_k",
    nullptr
};

// Conversion context structure
struct gguf_conversion_ctx {
    gguf_conversion_params_t params;
    std::atomic<float> progress;
    std::atomic<bool> cancelled;
    std::string current_stage;
    std::mutex stage_mutex;
    
    gguf_conversion_ctx() : progress(0.0f), cancelled(false) {}
    
    void set_progress(float p, const char* stage) {
        progress.store(p);
        {
            std::lock_guard<std::mutex> lock(stage_mutex);
            if (stage) {
                current_stage = stage;
            }
        }
        if (params.progress_cb) {
            params.progress_cb(p, stage, params.user_data);
        }
    }
    
    void log(int level, const std::string& message) {
        if (params.log_cb) {
            params.log_cb(level, message.c_str(), params.user_data);
        }
    }
};

// Helper to set error message
static void set_error(const std::string& msg) {
    g_last_error = msg;
}

// Helper to detect model format from file structure
static std::string detect_format_impl(const fs::path& path) {
    if (!fs::exists(path)) {
        return "";
    }
    
    // Check if it's a GGUF file
    if (path.extension() == ".gguf") {
        return "gguf";
    }
    
    // Check for directory-based formats
    if (fs::is_directory(path)) {
        // PyTorch format
        if (fs::exists(path / "pytorch_model.bin") || 
            fs::exists(path / "model.safetensors") ||
            fs::exists(path / "pytorch_model-00001-of-00002.bin")) {
            
            if (fs::exists(path / "model.safetensors")) {
                return "safetensors";
            }
            return "pytorch";
        }
        
        // TensorFlow format
        if (fs::exists(path / "saved_model.pb") ||
            fs::exists(path / "tf_model.h5")) {
            return "tensorflow";
        }
        
        // JAX/Flax format
        if (fs::exists(path / "flax_model.msgpack")) {
            return "flax";
        }
    }
    
    // Check file extensions
    std::string ext = path.extension().string();
    if (ext == ".bin" || ext == ".pt" || ext == ".pth") {
        return "pytorch";
    }
    if (ext == ".safetensors") {
        return "safetensors";
    }
    if (ext == ".h5" || ext == ".pb") {
        return "tensorflow";
    }
    if (ext == ".msgpack") {
        return "flax";
    }
    
    return "";
}

// Helper to extract model info from config.json
static bool extract_model_info(const fs::path& path, gguf_model_info_t* info) {
    fs::path config_path;
    
    if (fs::is_directory(path)) {
        config_path = path / "config.json";
    } else {
        config_path = path.parent_path() / "config.json";
    }
    
    if (!fs::exists(config_path)) {
        return false;
    }
    
    std::ifstream file(config_path);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple JSON parsing for key fields (production code should use proper JSON library)
    std::string line;
    std::map<std::string, std::string> config;
    
    while (std::getline(file, line)) {
        // Extract key-value pairs (simplified)
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            
            // Remove quotes, whitespace, commas
            key.erase(remove_if(key.begin(), key.end(), 
                [](char c) { return c == '"' || c == ' ' || c == '\t'; }), key.end());
            value.erase(remove_if(value.begin(), value.end(), 
                [](char c) { return c == '"' || c == ',' || c == ' ' || c == '\t'; }), value.end());
            
            config[key] = value;
        }
    }
    
    // Extract relevant fields
    if (config.count("model_type")) {
        strncpy(info->model_type, config["model_type"].c_str(), sizeof(info->model_type) - 1);
    }
    if (config.count("architectures")) {
        strncpy(info->architecture, config["architectures"].c_str(), sizeof(info->architecture) - 1);
    }
    if (config.count("hidden_size")) {
        info->hidden_size = std::stoul(config["hidden_size"]);
    }
    if (config.count("num_hidden_layers")) {
        info->num_layers = std::stoul(config["num_hidden_layers"]);
    }
    if (config.count("vocab_size")) {
        info->vocab_size = std::stoul(config["vocab_size"]);
    }
    if (config.count("max_position_embeddings")) {
        info->context_length = std::stoul(config["max_position_embeddings"]);
    }
    
    return true;
}

// ============================================================================
// Public API Implementation
// ============================================================================

const char* gguf_version(void) {
    return GGUF_BRIDGE_VERSION;
}

const char* gguf_get_last_error(void) {
    return g_last_error.c_str();
}

void gguf_clear_error(void) {
    g_last_error.clear();
}

void gguf_default_params(gguf_conversion_params_t* params) {
    if (!params) return;
    
    memset(params, 0, sizeof(gguf_conversion_params_t));
    params->quantization = "f16";
    params->vocab_only = 0;
    params->use_mmap = 1;
    params->num_threads = 0; // Auto-detect
    params->pad_vocab = 0;
}

gguf_ctx_t gguf_create_context(const gguf_conversion_params_t* params) {
    if (!params || !params->input_path || !params->output_path) {
        set_error("Invalid parameters: input_path and output_path are required");
        return nullptr;
    }
    
    // Validate paths
    if (!fs::exists(params->input_path)) {
        set_error(std::string("Input path not found: ") + params->input_path);
        return nullptr;
    }
    
    // Create context
    auto ctx = new gguf_conversion_ctx();
    ctx->params = *params;
    
    // Deep copy string pointers
    if (params->input_path) {
        ctx->params.input_path = strdup(params->input_path);
    }
    if (params->output_path) {
        ctx->params.output_path = strdup(params->output_path);
    }
    if (params->model_type) {
        ctx->params.model_type = strdup(params->model_type);
    }
    if (params->quantization) {
        ctx->params.quantization = strdup(params->quantization);
    }
    if (params->vocab_type) {
        ctx->params.vocab_type = strdup(params->vocab_type);
    }
    
    ctx->log(1, std::string("Created conversion context: ") + params->input_path + 
             " -> " + params->output_path);
    
    return ctx;
}

int gguf_validate_input(gguf_ctx_t ctx, gguf_model_info_t* info) {
    if (!ctx) {
        set_error("Invalid context");
        return GGUF_ERROR_INVALID_ARGS;
    }
    
    fs::path input_path(ctx->params.input_path);
    
    // Detect format
    std::string format = detect_format_impl(input_path);
    if (format.empty()) {
        set_error("Could not detect model format from: " + std::string(ctx->params.input_path));
        return GGUF_ERROR_INVALID_FORMAT;
    }
    
    ctx->log(1, "Detected format: " + format);
    
    // Extract model info if requested
    if (info) {
        memset(info, 0, sizeof(gguf_model_info_t));
        
        if (!extract_model_info(input_path, info)) {
            ctx->log(2, "Warning: Could not extract full model info from config");
        }
        
        // Get file size
        try {
            if (fs::is_directory(input_path)) {
                uint64_t total_size = 0;
                for (const auto& entry : fs::recursive_directory_iterator(input_path)) {
                    if (fs::is_regular_file(entry)) {
                        total_size += fs::file_size(entry);
                    }
                }
                info->file_size = total_size;
            } else {
                info->file_size = fs::file_size(input_path);
            }
        } catch (const std::exception& e) {
            ctx->log(2, std::string("Warning: Could not determine file size: ") + e.what());
        }
    }
    
    return GGUF_SUCCESS;
}

int gguf_convert(gguf_ctx_t ctx) {
    if (!ctx) {
        set_error("Invalid context");
        return GGUF_ERROR_INVALID_ARGS;
    }

    try {
        ctx->set_progress(0.0f, "Initializing conversion");
        ctx->log(1, "Starting conversion...");

        // Check for cancellation
        if (ctx->cancelled.load()) {
            set_error("Conversion cancelled");
            return GGUF_ERROR_CANCELLED;
        }

        // Validate quantization type
        bool valid_quant = false;
        for (int i = 0; QUANTIZATION_TYPES[i] != nullptr; i++) {
            if (strcmp(ctx->params.quantization, QUANTIZATION_TYPES[i]) == 0) {
                valid_quant = true;
                break;
            }
        }
        if (!valid_quant) {
            set_error(std::string("Invalid quantization type: ") + ctx->params.quantization);
            return GGUF_ERROR_INVALID_QUANTIZATION;
        }

        ctx->set_progress(0.1f, "Loading model");

        // Initialize llama.cpp backend
        llama_backend_init();

        std::string input_path(ctx->params.input_path);
        std::string output_path(ctx->params.output_path);
        std::string quantization(ctx->params.quantization);
        
        // Determine if we're converting from HF format to GGUF or quantizing an existing GGUF
        std::string input_ext = fs::path(input_path).extension().string();
        
        if (fs::is_directory(input_path) || input_ext == ".bin" || input_ext == ".safetensors") {
            // Converting from Hugging Face format to GGUF
            // This requires calling the Python conversion script
            
            ctx->set_progress(0.2f, "Preparing conversion parameters");
            
            // Build command to call the Python conversion script
            std::string cmd = "python3 ";
            cmd += fs::path(__FILE__).parent_path().string() + "/../../vendor/llama-cpp/llama.cpp/convert_hf_to_gguf.py ";
            cmd += "\"" + input_path + "\" ";
            cmd += "--outfile \"" + output_path + "\" ";
            
            // Add quantization parameter
            cmd += "--outtype " + quantization + " ";
            
            if (ctx->params.vocab_only) {
                cmd += "--vocab-only ";
            }
            
            if (ctx->params.model_type) {
                cmd += "--model-type " + std::string(ctx->params.model_type) + " ";
            }
            
            if (ctx->params.num_threads > 0) {
                cmd += "--threads " + std::to_string(ctx->params.num_threads) + " ";
            }
            
            // Execute the conversion command
            ctx->set_progress(0.3f, "Executing conversion script");
            ctx->log(1, "Running: " + cmd);
            
            int result = system(cmd.c_str());
            
            if (result != 0) {
                set_error("Python conversion script failed with code: " + std::to_string(result));
                llama_backend_free();
                return GGUF_ERROR_CONVERSION_FAILED;
            }
            
        } else if (input_ext == ".gguf") {
            // Quantizing an existing GGUF file using llama.cpp's quantization function
            ctx->set_progress(0.2f, "Preparing quantization parameters");
            
            // Map quantization string to llama_ftype
            enum llama_ftype target_ftype = LLAMA_FTYPE_MOSTLY_F16; // default
            
            if (quantization == "f32") target_ftype = LLAMA_FTYPE_ALL_F32;
            else if (quantization == "f16") target_ftype = LLAMA_FTYPE_MOSTLY_F16;
            else if (quantization == "q8_0") target_ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
            else if (quantization == "q4_0") target_ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
            else if (quantization == "q4_1") target_ftype = LLAMA_FTYPE_MOSTLY_Q4_1;
            else if (quantization == "q5_0") target_ftype = LLAMA_FTYPE_MOSTLY_Q5_0;
            else if (quantization == "q5_1") target_ftype = LLAMA_FTYPE_MOSTLY_Q5_1;
            else if (quantization == "q2_k") target_ftype = LLAMA_FTYPE_MOSTLY_Q2_K;
            else if (quantization == "q3_k_s") target_ftype = LLAMA_FTYPE_MOSTLY_Q3_K_S;
            else if (quantization == "q3_k_m") target_ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;
            else if (quantization == "q3_k_l") target_ftype = LLAMA_FTYPE_MOSTLY_Q3_K_L;
            else if (quantization == "q4_k_s") target_ftype = LLAMA_FTYPE_MOSTLY_Q4_K_S;
            else if (quantization == "q4_k_m") target_ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
            else if (quantization == "q5_k_s") target_ftype = LLAMA_FTYPE_MOSTLY_Q5_K_S;
            else if (quantization == "q5_k_m") target_ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;
            else if (quantization == "q6_k") target_ftype = LLAMA_FTYPE_MOSTLY_Q6_K;
            
            // Prepare quantization parameters
            struct llama_model_quantize_params params = llama_model_quantize_default_params();
            params.ftype = target_ftype;
            params.nthread = ctx->params.num_threads;
            
            // Perform the quantization
            ctx->set_progress(0.3f, "Starting quantization process");
            uint32_t result = llama_model_quantize(
                input_path.c_str(),
                output_path.c_str(),
                &params
            );
            
            if (result != 0) {
                set_error("llama.cpp quantization failed with code: " + std::to_string(result));
                llama_backend_free();
                return GGUF_ERROR_CONVERSION_FAILED;
            }
        } else {
            set_error("Unsupported input format: " + input_ext);
            llama_backend_free();
            return GGUF_ERROR_INVALID_FORMAT;
        }
        
        ctx->set_progress(1.0f, "Complete");
        ctx->log(1, "Conversion completed successfully");

        llama_backend_free();

        return GGUF_SUCCESS;

    } catch (const std::exception& e) {
        set_error(std::string("Conversion error: ") + e.what());
        ctx->log(3, g_last_error);
        llama_backend_free();
        return GGUF_ERROR_CONVERSION_FAILED;
    }
}

void gguf_cancel(gguf_ctx_t ctx) {
    if (ctx) {
        ctx->cancelled.store(true);
        ctx->log(2, "Cancellation requested");
    }
}

int gguf_is_cancelled(gguf_ctx_t ctx) {
    return ctx ? ctx->cancelled.load() : 0;
}

float gguf_get_progress(gguf_ctx_t ctx) {
    return ctx ? ctx->progress.load() : -1.0f;
}

void gguf_free_context(gguf_ctx_t ctx) {
    if (!ctx) return;
    
    // Free duplicated strings
    if (ctx->params.input_path) free(const_cast<char*>(ctx->params.input_path));
    if (ctx->params.output_path) free(const_cast<char*>(ctx->params.output_path));
    if (ctx->params.model_type) free(const_cast<char*>(ctx->params.model_type));
    if (ctx->params.quantization) free(const_cast<char*>(ctx->params.quantization));
    if (ctx->params.vocab_type) free(const_cast<char*>(ctx->params.vocab_type));
    
    delete ctx;
}

const char* gguf_detect_format(const char* path) {
    if (!path) return nullptr;
    
    static thread_local std::string format_str;
    format_str = detect_format_impl(path);
    
    return format_str.empty() ? nullptr : format_str.c_str();
}

const char** gguf_available_quantizations(void) {
    return QUANTIZATION_TYPES;
}

int gguf_verify_file(const char* path, gguf_model_info_t* info) {
    if (!path) {
        set_error("Invalid path");
        return GGUF_ERROR_INVALID_ARGS;
    }
    
    fs::path file_path(path);
    if (!fs::exists(file_path)) {
        set_error("File not found");
        return GGUF_ERROR_FILE_NOT_FOUND;
    }
    
    // Open and verify GGUF magic number
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error("Could not open file");
        return GGUF_ERROR_IO_ERROR;
    }
    
    // Read magic number (GGUF)
    char magic[4];
    file.read(magic, 4);
    
    if (strncmp(magic, "GGUF", 4) != 0) {
        set_error("Invalid GGUF magic number");
        return GGUF_ERROR_INVALID_FORMAT;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (info) {
        memset(info, 0, sizeof(gguf_model_info_t));
        info->file_size = fs::file_size(file_path);
        strncpy(info->model_type, "gguf", sizeof(info->model_type) - 1);
        
        // Here you would parse the GGUF metadata to extract full model info
        // This requires understanding the GGUF format specification
    }
    
    return GGUF_SUCCESS;
}
