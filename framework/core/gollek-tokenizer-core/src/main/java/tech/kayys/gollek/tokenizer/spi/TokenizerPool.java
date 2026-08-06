package tech.kayys.gollek.tokenizer.runtime;

import tech.kayys.gollek.tokenizer.impl.FastSentencePieceTokenizer;

import java.nio.file.Path;

public class TokenizerPool {

    private final ThreadLocal<FastSentencePieceTokenizer> local;

    public TokenizerPool(Path lib, Path model) {
        this.local = ThreadLocal.withInitial(() -> new FastSentencePieceTokenizer(lib, model));
    }

    public FastSentencePieceTokenizer current() {
        return local.get();
    }

    public void shutdown() {
        local.get().close();
    }
}