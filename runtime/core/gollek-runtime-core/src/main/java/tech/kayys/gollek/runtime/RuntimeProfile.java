package tech.kayys.gollek.runtime;

public final class RuntimeProfile {
    public final boolean enableOnnx;
    public final boolean enableGGUF;
    public final boolean enableQuant;
    public final boolean enableFlashAttention;

    public RuntimeProfile(
            boolean onnx,
            boolean gguf,
            boolean quant,
            boolean flash) {
        this.enableOnnx = onnx;
        this.enableGGUF = gguf;
        this.enableQuant = quant;
        this.enableFlashAttention = flash;
    }
}