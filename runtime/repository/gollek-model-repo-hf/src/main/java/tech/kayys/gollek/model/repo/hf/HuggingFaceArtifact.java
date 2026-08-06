package tech.kayys.gollek.model.repo.hf;

import java.net.URI;


/**
 * Details of a resolved HuggingFace artifact
 */
public record HuggingFaceArtifact(
    String id,
    String repo,
    String revision,
    String filename,
    String format,
    URI downloadUri
) {}
