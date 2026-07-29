package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.spec.*;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

/**
 * Supported FFN activation functions.
 */
public enum FFNActivationType {
    SILU, // LLaMA, Mistral
    GELU, // Gemma, Gemma-2
    RELU,
    GELU_QUICK
}
