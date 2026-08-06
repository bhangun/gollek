package tech.kayys.gollek.safetensor.engine.route;

import java.nio.file.Path;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

public interface LocalModelResolver {

    public interface LocalModelCandidate {
        public String format();
        public String quantization();
        public String architecture();
        public String provider();
        public String id();
        public String name();
        public String shortId();
        public Path path();
    }

    public Stream<LocalModelCandidate> localModels();

    public Optional<Path> resolveLiteRtModel(String modelId);
}
