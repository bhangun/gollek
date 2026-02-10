package tech.kayys.golek.spi.model;

import java.util.Map;

public record ModelRef(
        String scheme, // hf, local, s3, git, http, custom
        String namespace, // org/user
        String name,
        String version,
        Map<String, String> parameters) {
}
