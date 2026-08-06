package tech.kayys.gollek.tokenizer.impl;

import java.util.List;
import tech.kayys.gollek.tokenizer.spi.PreTokenizer;

/**
 * SentencePiece-style pretokenizer.
 *
 * <p>Normalizes spaces into the "▁" marker and returns the
 * whole string as a single token piece for BPE.</p>
 */
public final class SentencePiecePreTokenizer implements PreTokenizer {
    @Override
    public List<String> split(String text) {
        if (text == null || text.isEmpty()) {
            return List.of("");
        }
        String normalized = text.replace(' ', '▁');
        return List.of(normalized);
    }
}
