package tech.kayys.golek.model.core;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;

public record RepositoryContext(
                Path cacheDir,
                Duration timeout,
                Map<String, Object> attributes) {
}