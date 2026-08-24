/**
 * GGUF Quantizer Implementation
 * Implements K-quant and standard quantization algorithms
 */

#include "gguf_quantizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

// Helper functions
static inline uint16_t float_to_half(float f) {
  uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
  uint32_t sign = (bits >> 31) & 0x1;
  uint32_t exp = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x7FFFFF;

  if (exp == 0xFF) {
    // Infinity or NaN
    return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
  }

  int32_t new_exp = (int32_t)exp - 127 + 15;
  if (new_exp <= 0) {
    // Subnormal or zero
    if (new_exp < -10)
      return sign << 15;
    mant = (mant | 0x800000) >> (1 - new_exp);
    return (sign << 15) | ((mant + 0x1000) >> 13);
  }

  if (new_exp >= 0x1F) {
    return (sign << 15) | 0x7C00;
  }

  return (sign << 15) | (new_exp << 10) | (mant >> 13);
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    // Subnormal
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    uint32_t value = (sign << 31) | (mant << 13);
    return *reinterpret_cast<float *>(&value);
  }

  if (exp == 0x1F) {
    // Infinity or NaN
    uint32_t value = (sign << 31) | 0x7F800000 | (mant << 13);
    return *reinterpret_cast<float *>(&value);
  }

  uint32_t value = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  return *reinterpret_cast<float *>(&value);
}

static inline int nearest_int(float x) {
  if (x >= 0) {
    return (int)(x + 0.5f);
  } else {
    return (int)(x - 0.5f);
  }
}

void quantize_f32_to_f16(const float *src, uint16_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = float_to_half(src[i]);
  }
}

void quantize_f32_to_q4_0(const float *src, uint8_t *dst, size_t n) {
  // Block size: 32 elements
  size_t num_blocks = n / 32;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 32;
    uint8_t *block_dst = dst + b * 18;

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
      max_abs = std::max(max_abs, std::abs(block_src[i]));
    }

    float scale = max_abs / 7.0f;
    float inv_scale = scale == 0.0f ? 0.0f : 1.0f / scale;

    // Store scale as F16
    uint16_t *scale_ptr = reinterpret_cast<uint16_t *>(block_dst);
    *scale_ptr = float_to_half(scale);

    // Pack 4-bit weights (2 per byte)
    for (int i = 0; i < 16; i++) {
      float v0 = block_src[i * 2];
      float v1 = block_src[i * 2 + 1];

      int q0 = nearest_int(v0 * inv_scale) + 8;
      int q1 = nearest_int(v1 * inv_scale) + 8;

      q0 = std::max(0, std::min(15, q0));
      q1 = std::max(0, std::min(15, q1));

      block_dst[2 + i] = (q0) | (q1 << 4);
    }
  }
}

void quantize_f32_to_q4_1(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 32;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 32;
    uint8_t *block_dst = dst + b * 20;

    // Find min and max
    float min_val = block_src[0];
    float max_val = block_src[0];
    for (int i = 1; i < 32; i++) {
      min_val = std::min(min_val, block_src[i]);
      max_val = std::max(max_val, block_src[i]);
    }

    float scale = (max_val - min_val) / 15.0f;
    float inv_scale = scale == 0.0f ? 0.0f : 1.0f / scale;

    // Store min and scale as F16
    uint16_t *min_ptr = reinterpret_cast<uint16_t *>(block_dst);
    uint16_t *scale_ptr = reinterpret_cast<uint16_t *>(block_dst + 2);
    *min_ptr = float_to_half(min_val);
    *scale_ptr = float_to_half(scale);

    // Pack 4-bit weights (2 per byte)
    for (int i = 0; i < 16; i++) {
      float v0 = block_src[i * 2];
      float v1 = block_src[i * 2 + 1];

      int q0 = nearest_int((v0 - min_val) * inv_scale);
      int q1 = nearest_int((v1 - min_val) * inv_scale);

      q0 = std::max(0, std::min(15, q0));
      q1 = std::max(0, std::min(15, q1));

      block_dst[4 + i] = (q0) | (q1 << 4);
    }
  }
}

void quantize_f32_to_q8_0(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 32;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 32;
    uint8_t *block_dst = dst + b * 34;

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
      max_abs = std::max(max_abs, std::abs(block_src[i]));
    }

    float scale = max_abs / 127.0f;
    float inv_scale = scale == 0.0f ? 0.0f : 1.0f / scale;

    // Store scale as F16
    uint16_t *scale_ptr = reinterpret_cast<uint16_t *>(block_dst);
    *scale_ptr = float_to_half(scale);

    // Store 8-bit weights
    for (int i = 0; i < 32; i++) {
      int q = nearest_int(block_src[i] * inv_scale);
      q = std::max(-127, std::min(127, q));
      block_dst[2 + i] = (uint8_t)(q + 128);
    }
  }
}

void quantize_f32_to_q2_k(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 256;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 256;
    uint8_t *block_dst = dst + b * 84;

    // Process 16 sub-blocks of 16 elements each
    float scales[16];
    float mins[16];

    for (int sb = 0; sb < 16; sb++) {
      float min_val = block_src[sb * 16];
      float max_val = block_src[sb * 16];
      for (int i = 1; i < 16; i++) {
        float v = block_src[sb * 16 + i];
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
      }
      scales[sb] = (max_val - min_val) / 3.0f;
      mins[sb] = min_val;
    }

    // Pack scales and mins (4 bits each)
    for (int sb = 0; sb < 16; sb += 2) {
      int scale_byte = (std::min(15, (int)(scales[sb] * 15.0f / 7.0f)) << 4) |
                       std::min(15, (int)(scales[sb + 1] * 15.0f / 7.0f));
      int min_byte = (std::min(15, (int)(mins[sb] * 15.0f / 7.0f)) << 4) |
                     std::min(15, (int)(mins[sb + 1] * 15.0f / 7.0f));
      block_dst[sb / 2] = scale_byte;
      block_dst[8 + sb / 2] = min_byte;
    }

    // Pack 2-bit weights (4 per byte)
    for (int sb = 0; sb < 16; sb++) {
      float scale = scales[sb];
      float min_val = mins[sb];
      if (scale == 0.0f)
        scale = 1e-5f;

      for (int i = 0; i < 16; i += 4) {
        int packed = 0;
        for (int j = 0; j < 4; j++) {
          float v = block_src[sb * 16 + i + j];
          int q = (int)((v - min_val) / scale);
          q = std::max(0, std::min(3, q));
          packed |= (q << (j * 2));
        }
        block_dst[16 + sb * 2 + i / 4] = packed;
      }
    }
  }
}

void quantize_f32_to_q4_k(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 256;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 256;
    uint8_t *block_dst = dst + b * 144;

    // Process 8 sub-blocks of 32 elements each
    float scales[8];

    for (int sb = 0; sb < 8; sb++) {
      float max_abs = 0.0f;
      for (int i = 0; i < 32; i++) {
        max_abs = std::max(max_abs, std::abs(block_src[sb * 32 + i]));
      }
      scales[sb] = max_abs / 7.0f;
    }

    // Pack scales into 6 bits each (48 bits = 6 bytes)
    uint64_t packed_scales = 0;
    for (int sb = 0; sb < 8; sb++) {
      int q = std::min(63, (int)(scales[sb] * 63.0f / 7.0f));
      packed_scales |= ((uint64_t)q) << (sb * 6);
    }
    memcpy(block_dst, &packed_scales, 6);

    // Pack 4-bit weights (2 per byte)
    for (int sb = 0; sb < 8; sb++) {
      float scale = scales[sb];
      if (scale == 0.0f)
        scale = 1e-5f;
      float inv_scale = 1.0f / scale;

      for (int i = 0; i < 32; i += 2) {
        int q0 = nearest_int(block_src[sb * 32 + i] * inv_scale) + 8;
        int q1 = nearest_int(block_src[sb * 32 + i + 1] * inv_scale) + 8;
        q0 = std::max(0, std::min(15, q0));
        q1 = std::max(0, std::min(15, q1));
        block_dst[6 + sb * 16 + i / 2] = (q0) | (q1 << 4);
      }
    }
  }
}

void quantize_f32_to_q5_k(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 256;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 256;
    uint8_t *block_dst = dst + b * 176;

    // Process 8 sub-blocks of 32 elements each
    float scales[8];

    for (int sb = 0; sb < 8; sb++) {
      float max_abs = 0.0f;
      for (int i = 0; i < 32; i++) {
        max_abs = std::max(max_abs, std::abs(block_src[sb * 32 + i]));
      }
      scales[sb] = max_abs / 15.0f;
    }

    // Pack scales into 6 bits each
    uint64_t packed_scales = 0;
    for (int sb = 0; sb < 8; sb++) {
      int q = std::min(63, (int)(scales[sb] * 63.0f / 15.0f));
      packed_scales |= ((uint64_t)q) << (sb * 6);
    }
    memcpy(block_dst, &packed_scales, 6);

    // Store low and high bits separately
    uint8_t low_bits[128];
    uint8_t high_bits[32] = {0};

    for (int sb = 0; sb < 8; sb++) {
      float scale = scales[sb];
      if (scale == 0.0f)
        scale = 1e-5f;
      float inv_scale = 1.0f / scale;

      for (int i = 0; i < 32; i++) {
        int q = nearest_int(block_src[sb * 32 + i] * inv_scale) + 16;
        q = std::max(0, std::min(31, q));

        int idx = sb * 32 + i;
        low_bits[idx] = q & 0x0F;
        if (q & 0x10) {
          high_bits[idx / 8] |= (1 << (idx % 8));
        }
      }
    }

    memcpy(block_dst + 6, low_bits, 128);
    memcpy(block_dst + 134, high_bits, 32);
  }
}

void quantize_f32_to_q6_k(const float *src, uint8_t *dst, size_t n) {
  size_t num_blocks = n / 256;

  for (size_t b = 0; b < num_blocks; b++) {
    const float *block_src = src + b * 256;
    uint8_t *block_dst = dst + b * 210;

    // Process 16 sub-blocks of 16 elements each
    float scales[16];

    for (int sb = 0; sb < 16; sb++) {
      float max_abs = 0.0f;
      for (int i = 0; i < 16; i++) {
        max_abs = std::max(max_abs, std::abs(block_src[sb * 16 + i]));
      }
      scales[sb] = max_abs / 31.0f;
    }

    // Pack scales into 4 bits each (64 bits = 8 bytes)
    for (int sb = 0; sb < 16; sb += 2) {
      int q0 = std::min(15, (int)(scales[sb] * 15.0f / 31.0f));
      int q1 = std::min(15, (int)(scales[sb + 1] * 15.0f / 31.0f));
      block_dst[sb / 2] = (q0 << 4) | q1;
    }

    // Store low and high bits
    uint8_t low_bits[192];
    uint8_t high_bits[64] = {0};

    for (int sb = 0; sb < 16; sb++) {
      float scale = scales[sb];
      if (scale == 0.0f)
        scale = 1e-5f;
      float inv_scale = 1.0f / scale;

      for (int i = 0; i < 16; i++) {
        int q = nearest_int(block_src[sb * 16 + i] * inv_scale) + 32;
        q = std::max(0, std::min(63, q));

        int idx = sb * 16 + i;
        low_bits[idx] = q & 0x0F;
        high_bits[idx / 4] |= ((q >> 4) & 0x03) << ((idx % 4) * 2);
      }
    }

    memcpy(block_dst + 8, low_bits, 192);
    memcpy(block_dst + 200, high_bits, 64);
  }
}

void dequantize_f16_to_f32(const uint16_t *src, float *dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = half_to_float(src[i]);
  }
}

void dequantize_q4_0_to_f32(const uint8_t *src, float *dst, size_t n) {
  size_t num_blocks = n / 32;

  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t *block_src = src + b * 18;
    float *block_dst = dst + b * 32;

    float scale = half_to_float(*reinterpret_cast<const uint16_t *>(block_src));

    for (int i = 0; i < 16; i++) {
      uint8_t packed = block_src[2 + i];
      int q0 = packed & 0x0F;
      int q1 = (packed >> 4) & 0x0F;

      block_dst[i * 2] = (q0 - 8) * scale;
      block_dst[i * 2 + 1] = (q1 - 8) * scale;
    }
  }
}

void quantize_with_imatrix(const float *src, uint8_t *dst, size_t n,
                           const float *importance, int quant_type) {
  // For production, implement weighted Lloyd's algorithm
  // This is a placeholder that uses standard quantization

  switch (quant_type) {
  case 0: // Q4_K
    quantize_f32_to_q4_k(src, dst, n);
    break;
  case 1: // Q5_K
    quantize_f32_to_q5_k(src, dst, n);
    break;
  case 2: // Q6_K
    quantize_f32_to_q6_k(src, dst, n);
    break;
  default:
    quantize_f32_to_q4_k(src, dst, n);
    break;
  }
}