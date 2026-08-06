package tech.kayys.gollek.tokenizer.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.tokenizer.impl.BpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.Gpt2PreTokenizer;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.File;
import java.nio.file.Path;
import java.util.*;

public class HuggingFaceBpeLoader {

    public Tokenizer load(Path path) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);

            Map<String, Integer> tokenToId = json.model.vocab;
            Map<Integer, String> idToToken = new HashMap<>();
            tokenToId.forEach((k, v) -> idToToken.put(v, k));

            Map<String, Integer> mergeRanks = new HashMap<>();
            for (int i = 0; i < json.model.merges.size(); i++) {
                String[] parts = json.model.merges.get(i).split(" ");
                mergeRanks.put(parts[0] + parts[1], i);
            }

            return new BpeTokenizer(
                    tokenToId,
                    idToToken,
                    mergeRanks,
                    buildByteEncoder(),
                    findToken(json, "<s>"),
                    findToken(json, "</s>"),
                    findToken(json, "<pad>"),
                    findToken(json, "<unk>"),
                    new Gpt2PreTokenizer());

        } catch (Exception e) {
            throw new RuntimeException("Failed to load tokenizer", e);
        }
    }

    private int findToken(TokenizerJson json, String token) {
        return json.model.vocab.getOrDefault(token, -1);
    }

    private Map<Character, String> buildByteEncoder() {
        Map<Character, String> encoder = new HashMap<>();
        for (int i = 0; i < 256; i++) {
            encoder.put((char) i, String.valueOf((char) i));
        }
        return encoder;
    }

    public static class TokenizerJson {
        public Model model;

        public static class Model {
            public Map<String, Integer> vocab;
            public List<String> merges;
        }
    }
}