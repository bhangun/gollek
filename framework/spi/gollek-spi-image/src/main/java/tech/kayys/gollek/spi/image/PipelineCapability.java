package tech.kayys.gollek.spi.image;

public enum PipelineCapability {
    TEXT_TO_IMAGE,
    IMAGE_TO_IMAGE,
    INPAINTING,
    OUTPAINTING,
    UPSCALING,
    CONTROLNET,
    IP_ADAPTER;

    public boolean isGeneration() {
        return this == TEXT_TO_IMAGE || this == IMAGE_TO_IMAGE;
    }

    public boolean isEditing() {
        return this == INPAINTING || this == OUTPAINTING;
    }
}
