package tech.kayys.gollek.tokenizer.runtime;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.tokenizer.impl.FastSentencePieceTokenizer;
import tech.kayys.gollek.tokenizer.impl.Gpt2PreTokenizer;
import tech.kayys.gollek.tokenizer.impl.HuggingFaceBpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.SentencePiecePreTokenizer;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.TokenizerType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Factory for creating Tokenizer instances based on model directory content.
 */
public class TokenizerFactory {

    /**
     * Automatically detect and load a tokenizer from the given model directory.
     *
     * @param modelDir the directory containing tokenizer files
     * @param nativeLibPath optional path to the native SPM bridge library (required for SentencePiece)
     * @return a configured Tokenizer instance
     * @throws IOException if files are missing or unreadable
     */
    public static Tokenizer load(Path modelDir, Path nativeLibPath) throws IOException {
        if (modelDir == null) {
            throw new IOException("Model path is null");
        }
        if (Files.isRegularFile(modelDir)) {
            Path parent = modelDir.getParent();
            if (parent == null) {
                throw new IOException("Model path has no parent directory: " + modelDir);
            }
            modelDir = parent;
        }
        Path hfConfig = modelDir.resolve("tokenizer.json");
        Path spmModel = modelDir.resolve("tokenizer.model");

        // Diffusers / Stable Diffusion compatibility: check 'tokenizer' subdirectory
        if (!Files.exists(hfConfig) && !Files.exists(spmModel)) {
            Path subDir = modelDir.resolve("tokenizer");
            if (Files.isDirectory(subDir)) {
                hfConfig = subDir.resolve("tokenizer.json");
                spmModel = subDir.resolve("tokenizer.model");
            }
        }

        if (Files.exists(spmModel)) {
            Path effectiveLibPath = nativeLibPath;
            if (effectiveLibPath == null) {
                effectiveLibPath = NativeDiscovery.findSpmBridge().orElse(null);
            }

            if (effectiveLibPath == null || !Files.exists(effectiveLibPath)) {
                throw new IOException("SentencePiece model found but native bridge library path is missing or invalid: " + spmModel.getParent() + "/libspm_bridge");
            }
            return new FastSentencePieceTokenizer(effectiveLibPath, spmModel);
        }

        if (Files.exists(hfConfig)) {
            String modelType = detectTokenizerModelType(hfConfig);
            if (modelType != null && !modelType.equalsIgnoreCase("BPE")) {
                throw new IOException("Unsupported tokenizer.json model type: " + modelType
                        + " (expected BPE). If this model uses SentencePiece, ensure tokenizer.model is present and the spm_bridge native library is installed.");
            }
            if (isSentencePieceStyle(hfConfig)) {
                return HuggingFaceBpeTokenizer.load(hfConfig, new SentencePiecePreTokenizer(), false, true);
            }
            // Default to GPT-2 pre-tokenizer for HuggingFace BPE models
            return HuggingFaceBpeTokenizer.load(hfConfig, new Gpt2PreTokenizer());
        }

        // Try vocab.json + merges.txt fallback (Legacy CLIP/BPE format)
        Path vocabJson = modelDir.resolve("vocab.json");
        Path mergesTxt = modelDir.resolve("merges.txt");

        // Diffusers / Stable Diffusion compatibility: check 'tokenizer' subdirectory
        if (!Files.exists(vocabJson) || !Files.exists(mergesTxt)) {
            Path subDir = modelDir.resolve("tokenizer");
            if (Files.isDirectory(subDir)) {
                vocabJson = subDir.resolve("vocab.json");
                mergesTxt = subDir.resolve("merges.txt");
            }
        }

        if (Files.exists(vocabJson) && Files.exists(mergesTxt)) {
            // Assume GPT-2/CLIP style BPE
            return HuggingFaceBpeTokenizer.load(vocabJson, mergesTxt, new Gpt2PreTokenizer(), true, false);
        }

        throw new IOException("No supported tokenizer files found in " + modelDir + " (expected tokenizer.json, tokenizer.model, or vocab.json+merges.txt)");
    }

    private static String detectTokenizerModelType(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());
            JsonNode model = root.path("model");
            if (model.isMissingNode()) {
                return null;
            }
            String type = model.path("type").asText();
            return type == null || type.isBlank() ? null : type;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Load a specific type of tokenizer.
     */
    public static Tokenizer load(Path path, TokenizerType type, Path nativeLibPath) throws IOException {
        return switch (type) {
            case BPE -> HuggingFaceBpeTokenizer.load(path, new Gpt2PreTokenizer());
            case SENTENCE_PIECE -> new FastSentencePieceTokenizer(nativeLibPath, path);
            default -> throw new UnsupportedOperationException("Tokenizer type not yet implemented: " + type);
        };
    }

    private static boolean isSentencePieceStyle(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());
            JsonNode pre = root.path("pre_tokenizer");
            JsonNode norm = root.path("normalizer");
            boolean splitOnSpace = "Split".equals(pre.path("type").asText())
                    && " ".equals(pre.path("pattern").path("String").asText());
            boolean normToUnderline = "Replace".equals(norm.path("type").asText())
                    && " ".equals(norm.path("pattern").path("String").asText())
                    && "▁".equals(norm.path("content").asText());
            return splitOnSpace && normToUnderline;
        } catch (Exception e) {
            return false;
        }
    }
}
