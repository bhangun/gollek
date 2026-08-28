package tech.kayys.gollek.diffusion.threed;

/**
 * Configuration parameters for the Block3D discrete block-wise diffusion framework
 * (arXiv:2608.19567v4).
 *
 * @param sequenceLength Total shape sequence length N (default: 1024).
 * @param codebookSize   Discrete VQ codebook size V (default: 16384).
 * @param blockSize      Contiguous block size B (default: 64, yielding K=16 blocks).
 * @param horizon        Denoising update horizon per block T (default: 4).
 * @param guidanceScale  Classifier-Free Guidance coefficient g (default: 3.0).
 * @param etaM           Confidence threshold for Mask-to-Token (M2T) filling (default: 0.60).
 * @param etaT           Confidence threshold for Token-to-Token (T2T) revision (default: 0.85).
 * @param maskTokenId    Identifier for special mask token [M] (default: 16384).
 */
public record Block3DConfig(
        int sequenceLength,
        int codebookSize,
        int blockSize,
        int horizon,
        double guidanceScale,
        double etaM,
        double etaT,
        int maskTokenId
) {
    public static Block3DConfig defaultConfig() {
        return new Block3DConfig(
                1024,   // N = 1024 discrete shape codes
                16384,  // V = 16384 codebook vocabulary
                64,     // B = 64 codes per block (K = 16 blocks)
                4,      // T = 4 denoising iterations per block
                3.0,    // g = 3.0 CFG scale
                0.60,   // eta_M = 0.60 M2T threshold
                0.85,   // eta_T = 0.85 T2T revision threshold
                16384   // [M] mask token ID
        );
    }

    public static Block3DConfig fastPreview() {
        return new Block3DConfig(
                256,
                4096,
                32,
                2,
                2.5,
                0.50,
                0.80,
                4096
        );
    }

    public int numBlocks() {
        return (int) Math.ceil((double) sequenceLength / blockSize);
    }
}
