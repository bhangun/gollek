package tech.kayys.golek.spi.model;

import java.net.URI;
import java.util.Map;

public record ModelDescriptor(
        String id,
        String format, // gguf, onnx, safetensors, triton, etc
        URI source,
        Map<String, String> metadata) {
}
