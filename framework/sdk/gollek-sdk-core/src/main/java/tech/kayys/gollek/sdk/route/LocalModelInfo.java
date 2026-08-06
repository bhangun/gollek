package tech.kayys.gollek.sdk.route;

import java.nio.file.Path;

public record LocalModelInfo(
        String id,
        String shortId,
        String name,
        String architecture,
        String parameterCount,
        String format,
        String quantization,
        Path path,
        String provider) {
}
