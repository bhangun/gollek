package tech.kayys.gollek.tokenizer;

import java.nio.file.Path;
import tech.kayys.gollek.tokenizer.impl.SentencePieceTokenizer;

public class SentencePieceTokenizerPool {

    private final ThreadLocal<SentencePieceTokenizer> local;

    public SentencePieceTokenizerPool(Path lib, Path model) {
        this.local = ThreadLocal.withInitial(() -> new SentencePieceTokenizer(lib, model));
    }

    public SentencePieceTokenizer current() {
        return local.get();
    }
}