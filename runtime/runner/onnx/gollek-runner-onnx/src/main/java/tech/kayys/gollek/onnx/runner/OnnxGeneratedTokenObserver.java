package tech.kayys.gollek.onnx.runner;

@FunctionalInterface
interface OnnxGeneratedTokenObserver {

    OnnxGeneratedTokenObserver NOOP = (tokenId, tokenIndex) -> null;

    String onToken(int tokenId, int tokenIndex);

    default String onToken(int tokenId, int tokenIndex, boolean currentTextNeeded) {
        return onToken(tokenId, tokenIndex);
    }
}
