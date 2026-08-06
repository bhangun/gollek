package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.spi.PreTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Regex-based pre-tokenizer for GPT-2 / LLaMA style tokenization.
 *
 * <p>Splits text into words/pieces using specific regex rules that handle
 * punctuation, numbers, and whitespace contractions (e.g., "'re", "'ve").
 */
public class Gpt2PreTokenizer implements PreTokenizer {

    private static final Pattern PRE_TOKENIZE_PATTERN = Pattern.compile(
            "'(?:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+");

    @Override
    public List<String> split(String text) {
        List<String> words = new ArrayList<>();
        Matcher matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        while (matcher.find()) {
            words.add(matcher.group());
        }
        if (words.isEmpty() && !text.isEmpty()) {
            words.add(text); // Fallback: single token
        }
        return words;
    }
}