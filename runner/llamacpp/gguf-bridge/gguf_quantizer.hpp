/**
 * GGUF Quantizer
 * Implements K-quant and standard quantization algorithms
 */

#ifndef GGUF_QUANTIZER_HPP
#define GGUF_QUANTIZER_HPP

#include <cstdint>
#include <vector>

// Quantization functions
void quantize_f32_to_f16(const float *src, uint16_t *dst, size_t n);
void quantize_f32_to_q4_0(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q4_1(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q5_0(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q5_1(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q8_0(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q2_k(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q3_k(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q4_k(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q5_k(const float *src, uint8_t *dst, size_t n);
void quantize_f32_to_q6_k(const float *src, uint8_t *dst, size_t n);

// Dequantization for verification
void dequantize_f16_to_f32(const uint16_t *src, float *dst, size_t n);
void dequantize_q4_0_to_f32(const uint8_t *src, float *dst, size_t n);
void dequantize_q4_1_to_f32(const uint8_t *src, float *dst, size_t n);

// Importance matrix guided quantization
void quantize_with_imatrix(const float *src, uint8_t *dst, size_t n,
                           const float *importance, int quant_type);

#endif // GGUF_QUANTIZER_HPP