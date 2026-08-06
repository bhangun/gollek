package tech.kayys.gollek.tokenizer.spi;

import java.nio.file.Path;

public class ModelConfig {

    private final TokenizerType tokenizerType;
    private final Path tokenizerPath;

    public ModelConfig(TokenizerType tokenizerType, Path tokenizerPath) {
        this.tokenizerType = tokenizerType;
        this.tokenizerPath = tokenizerPath;
    }

    public TokenizerType getTokenizerType() {
        return tokenizerType;
    }

    public Path getTokenizerPath() {
        return tokenizerPath;
    }
}