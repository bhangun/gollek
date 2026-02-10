package tech.kayys.golek.model.repo.hf;


import java.net.URI;
import tech.kayys.golek.spi.model.ModelRef;

/**
 * Helper to resolve HuggingFace refs into structured artifact info.
 */
public class HuggingFaceArtifactResolver {

    private final HuggingFaceClient client;

    public HuggingFaceArtifactResolver(HuggingFaceClient client) {
        this.client = client;
    }

    public HuggingFaceArtifact resolve(ModelRef ref) {
        String repo = (ref.namespace() != null ? ref.namespace() + "/" : "") + ref.name();
        String revision = ref.version() != null ? ref.version() : "main";
        String format = ref.parameters().getOrDefault("format", "auto");
        String filename = ref.parameters().get("filename");

        // Construct a download URI
        URI downloadUri = URI.create(String.format("https://huggingface.co/%s/resolve/%s/%s", 
                repo, revision, filename != null ? filename : ""));

        return new HuggingFaceArtifact(
                ref.name(),
                repo,
                revision,
                filename,
                format,
                downloadUri
        );
    }
}