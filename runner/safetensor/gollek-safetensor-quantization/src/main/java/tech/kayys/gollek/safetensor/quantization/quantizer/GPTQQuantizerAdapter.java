/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPTQQuantizerAdapter.java
 * ─────────────────────────
 * Adapter that bridges gollek-quantizer-gptq to the safetensor Quantizer SPI.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.gptq.GPTQQuantizerService;
import tech.kayys.gollek.quantizer.gptq.GPTQConfig;

import org.jboss.logging.Logger;

import java.nio.file.Path;

/**
 * Adapter for GPTQ quantization.
 * <p>
 * Wraps the GPTQ quantizer module to implement the Quantizer SPI,
 * enabling integration with the safetensor quantization pipeline.
 */
public class GPTQQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(GPTQQuantizerAdapter.class);
    private final GPTQQuantizerService service = new GPTQQuantizerService();

    @Override
    public TorchTensor quantizeTensor(TorchTensor tensor, QuantConfig config) {
        // TODO: Implement tensor-level GPTQ quantization
        // Current GPTQ implementation works at model level
        log.warn("GPTQ operates at model level, not tensor level. Use GPTQQuantizerService.quantize() for full model quantization.");
        return tensor;
    }

    @Override
    public TorchTensor dequantizeTensor(TorchTensor quantizedTensor, QuantConfig config) {
        // TODO: Implement tensor-level GPTQ dequantization
        log.warn("GPTQ operates at model level, not tensor level. Use GPTQDequantizer for full model dequantization.");
        return quantizedTensor;
    }

    @Override
    public String getName() {
        return "GPTQ";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy.INT4;
    }

    /**
     * Quantize a full model using GPTQ.
     *
     * @param modelPath  path to FP32 model
     * @param outputPath path for quantized output
     * @param config     GPTQ configuration
     * @return quantization result
     */
    public GPTQQuantizerService.QuantizationResult quantizeModel(Path modelPath, Path outputPath, GPTQConfig config) {
        try {
            return service.quantize(modelPath, outputPath, config);
        } catch (Exception e) {
            throw new RuntimeException("GPTQ model quantization failed", e);
        }
    }
}
