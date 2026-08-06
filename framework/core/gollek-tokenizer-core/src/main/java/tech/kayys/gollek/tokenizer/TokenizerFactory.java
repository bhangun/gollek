package tech.kayys.gollek.tokenizer;

import tech.kayys.gollek.tokenizer.loader.HuggingFaceBpeLoader;
import tech.kayys.gollek.tokenizer.spi.*;

public class TokenizerFactory {

    public static Tokenizer create(ModelConfig config) {

        return switch (config.getTokenizerType()) {

            case BPE -> new HuggingFaceBpeLoader()
                    .load(config.getTokenizerPath());

            case SENTENCE_PIECE -> {
                throw new UnsupportedOperationException("SentencePiece not yet implemented");
            }

            case WORD_PIECE -> {
                throw new UnsupportedOperationException("WordPiece not yet implemented");
            }

            case TIKTOKEN -> {
                throw new UnsupportedOperationException("Tiktoken not yet implemented");
            }
        };
    }
}