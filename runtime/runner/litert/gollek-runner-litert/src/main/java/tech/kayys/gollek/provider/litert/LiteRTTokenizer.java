package tech.kayys.gollek.provider.litert;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Tokenizer for LiteRT LLM models.
 *
 * <p>Supports:
 * <ul>
 *   <li>SentencePiece model (extracted from .task bundle)</li>
 *   <li>tokenizer.json (HuggingFace format)</li>
 *   <li>Byte-level fallback for basic operation</li>
 * </ul>
 *
 * <p>For Gemma 4 E2B models, the SPM vocabulary is embedded in the .task file
 * as a TFLite buffer referenced by the "spm_vocab_model" metadata entry.
 *
 * <p>Since we cannot use the SentencePiece C++ library directly from Java without
 * native bindings, we implement a simplified BPE-like tokenizer that reads
 * the SPM protobuf vocab. For production use with full SPM support, a native
 * SentencePiece binding (via FFM or JNI) would be needed.
 */
@SuppressWarnings("unused")
public class LiteRTTokenizer implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(LiteRTTokenizer.class);

    // Common Gemma token IDs
    public static final int PAD_TOKEN_ID = 0;
    public static final int BOS_TOKEN_ID = 2;
    public static final int EOS_TOKEN_ID = 1;
    public static final int UNK_TOKEN_ID = 3;

    // Token constants for Gemma 4 chat template
    private static final String USER_TURN = "<|turn>user";
    private static final String MODEL_TURN = "<|turn>model";
    private static final String END_OF_TURN = "<turn|>";

    private final Map<String, Integer> vocab = new HashMap<>();
    private final Map<Integer, String> reverseVocab = new HashMap<>();
    private final Map<String, Integer> specialTokens = new HashMap<>();
    private tech.kayys.gollek.tokenizer.spi.Tokenizer coreTokenizer;
    private int vocabSize = 0;
    private boolean initialized = false;
    private TokenizerType type = TokenizerType.BYTE_FALLBACK;

    enum TokenizerType {
        SPM_PROTO,       // SentencePiece protobuf model
        HF_JSON,         // HuggingFace tokenizer.json
        BYTE_FALLBACK    // Byte-level fallback
    }

    /**
     * Create a tokenizer from a model directory or file.
     */
    public static LiteRTTokenizer create(Path modelPath) {
        LiteRTTokenizer tokenizer = new LiteRTTokenizer();
        try {
            Path tokenizerSource = normalizeTokenizerSource(modelPath);

            // Prefer a real HuggingFace tokenizer.json when a companion model
            // exists. Gemma 4 checkpoints expose a BPE tokenizer here, and
            // forcing the embedded `spm_vocab_model` through SentencePiece
            // yields incorrect prompt/response tokenization.
            Path modelDir = Files.isDirectory(tokenizerSource) ? tokenizerSource : tokenizerSource.getParent();
            if (modelDir != null) {
                Path tokenizerJson = modelDir.resolve("tokenizer.json");
                if (Files.exists(tokenizerJson)) {
                    tokenizer.loadWithCoreTokenizer(tokenizerJson);
                    return tokenizer;
                }

                // Try tokenizer.model (SPM binary format)
                Path tokenizerModel = modelDir.resolve("tokenizer.model");
                if (Files.exists(tokenizerModel)) {
                    tokenizer.loadFromSpmFile(tokenizerModel);
                    return tokenizer;
                }
            }

            // Search companion blob/safetensor trees for tokenizer.json before
            // falling back to the embedded LiteRT metadata buffer.
            Optional<Path> companionTokenizer = findCompanionTokenizerJson(tokenizerSource);
            if (companionTokenizer.isPresent()) {
                tokenizer.loadWithCoreTokenizer(companionTokenizer.get());
                return tokenizer;
            }

            // Only fall back to the embedded SentencePiece model when no real
            // tokenizer.json or tokenizer.model is available.
            if (Files.isRegularFile(tokenizerSource)) {
                Optional<byte[]> spmData = LiteRTContainerParser.extractSpmVocab(tokenizerSource);
                if (spmData.isPresent() && spmData.get().length > 0) {
                    tokenizer.loadFromSpmBytes(spmData.get());
                    return tokenizer;
                }
            }

            // Fallback to byte-level tokenizer
            log.warn("No tokenizer found, using byte-level fallback. " +
                    "For better results, place tokenizer.json in the model directory.");
            tokenizer.initByteFallback();

        } catch (Exception e) {
            log.warn("Failed to load tokenizer: {}. Using byte-level fallback.", e.getMessage());
            tokenizer.initByteFallback();
        }
        return tokenizer;
    }

    private static Path normalizeTokenizerSource(Path modelPath) {
        try {
            return LiteRTContainerParser.findBestModelFile(modelPath).orElse(modelPath);
        } catch (Exception e) {
            return modelPath;
        }
    }

    /**
     * Encode text to token IDs.
     */
    public int[] encode(String text) {
        if (!initialized) {
            initByteFallback();
        }

        if (coreTokenizer != null) {
            return toIntArray(coreTokenizer.encode(text, EncodeOptions.defaultOptions()));
        }

        if (type == TokenizerType.BYTE_FALLBACK) {
            return encodeByteFallback(text);
        }

        return encodeWithVocab(text);
    }

    /**
     * Encode text with BOS token prepended.
     */
    public int[] encodeWithBos(String text) {
        if (coreTokenizer != null) {
            EncodeOptions options = EncodeOptions.builder()
                    .addBos(true)
                    .build();
            return toIntArray(coreTokenizer.encode(text, options));
        }
        int[] tokens = encode(text);
        int[] result = new int[tokens.length + 1];
        result[0] = BOS_TOKEN_ID;
        System.arraycopy(tokens, 0, result, 1, tokens.length);
        return result;
    }

    /**
     * Format a chat-style prompt using Gemma's template.
     */
    public int[] encodeChatPrompt(String userMessage) {
        // Gemma 4 chat format:
        // <bos><|turn>user\n{message}<turn|>\n<|turn>model\n
        String formatted = USER_TURN + "\n" + userMessage + END_OF_TURN +
                "\n" + MODEL_TURN + "\n";
        return encodeWithBos(formatted);
    }

    /**
     * Format a full conversation using Gemma 4's turn-based template.
     */
    public int[] encodeChatPrompt(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return encodeChatPrompt("");
        }

        StringBuilder formatted = new StringBuilder();
        String firstUserPrefix = "";
        int startIdx = 0;

        Message first = messages.get(0);
        if (first.getRole() == Message.Role.SYSTEM && first.getContent() != null) {
            firstUserPrefix = first.getContent().trim();
            if (!firstUserPrefix.isEmpty()) {
                firstUserPrefix += "\n\n";
            }
            startIdx = 1;
        }

        boolean firstRendered = true;
        for (int i = startIdx; i < messages.size(); i++) {
            Message message = messages.get(i);
            String content = message.getContent();
            if (content == null || content.isBlank()) {
                continue;
            }

            String roleName = switch (message.getRole()) {
                case ASSISTANT -> "model";
                case SYSTEM, USER -> "user";
                default -> "user";
            };

            formatted.append("<|turn>").append(roleName).append("\n");
            if (firstRendered && !firstUserPrefix.isEmpty() && message.getRole() == Message.Role.USER) {
                formatted.append(firstUserPrefix);
            }
            formatted.append(content.trim()).append("<turn|>\n");
            firstRendered = false;
        }

        formatted.append("<|turn>model\n");
        return encodeWithBos(formatted.toString());
    }

    /**
     * Decode token IDs to text.
     */
    public String decode(int[] tokenIds) {
        if (coreTokenizer != null) {
            return coreTokenizer.decode(toLongArray(tokenIds), DecodeOptions.defaultOptions());
        }
        if (type == TokenizerType.BYTE_FALLBACK) {
            return decodeByteFallback(tokenIds);
        }
        return decodeWithVocab(tokenIds);
    }

    /**
     * Decode a single token ID to text.
     */
    public String decodeToken(int tokenId) {
        if (tokenId == BOS_TOKEN_ID || tokenId == EOS_TOKEN_ID || tokenId == PAD_TOKEN_ID) {
            return "";
        }
        if (coreTokenizer != null) {
            DecodeOptions options = DecodeOptions.builder()
                    .skipSpecialTokens(false)
                    .build();
            return coreTokenizer.decode(new long[]{tokenId}, options);
        }
        String piece = reverseVocab.get(tokenId);
        if (piece == null) {
            return "";
        }
        // SentencePiece uses ▁ (U+2581) for space
        return piece.replace("▁", " ");
    }

    /**
     * Check if a token is an end-of-sequence token.
     */
    public boolean isEosToken(int tokenId) {
        if (coreTokenizer != null) {
            return tokenId == coreTokenizer.eosTokenId();
        }
        return tokenId == EOS_TOKEN_ID;
    }

    /**
     * End-of-turn and EOS markers should not be surfaced as user-visible text.
     */
    public boolean isTerminalToken(int tokenId) {
        if (coreTokenizer != null) {
            if (tokenId == coreTokenizer.eosTokenId()) {
                return true;
            }
            int[] stopTokenIds = coreTokenizer.allStopTokenIds();
            for (int stopTokenId : stopTokenIds) {
                if (tokenId == stopTokenId) {
                    return true;
                }
            }
        }
        if (isEosToken(tokenId)) {
            return true;
        }
        String piece = decodeToken(tokenId).trim();
        return "<turn|>".equals(piece)
                || "<end_of_turn>".equals(piece)
                || "<eos>".equals(piece);
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        if (coreTokenizer != null) {
            return coreTokenizer.vocabSize();
        }
        return vocabSize;
    }

    /**
     * Check if tokenizer is initialized with a real vocabulary (not byte fallback).
     */
    public boolean hasVocabulary() {
        return coreTokenizer != null || type != TokenizerType.BYTE_FALLBACK;
    }

    @Override
    public void close() {
        vocab.clear();
        reverseVocab.clear();
        specialTokens.clear();
        coreTokenizer = null;
        initialized = false;
    }

    private void loadWithCoreTokenizer(Path tokenizerPath) throws IOException {
        coreTokenizer = TokenizerFactory.load(tokenizerPath, null);
        specialTokens.clear();
        specialTokens.putAll(coreTokenizer.specialTokens());
        vocabSize = coreTokenizer.vocabSize();
        type = TokenizerType.HF_JSON;
        initialized = true;
        log.info("Loaded tokenizer via tokenizer-core from {} (vocab size: {})",
                tokenizerPath, vocabSize);
    }

    // ===== Internal: SPM Protobuf parsing =====

    private void loadFromSpmBytes(byte[] spmData) {
        try {
            parseSpmProtobuf(spmData);
            type = TokenizerType.SPM_PROTO;
            initialized = true;
            log.info("Loaded SPM tokenizer, vocab size: {}", vocabSize);
        } catch (Exception e) {
            log.warn("Failed to parse SPM protobuf: {}. Using byte fallback.", e.getMessage());
            initByteFallback();
        }
    }

    private void loadFromSpmFile(Path spmFile) throws IOException {
        byte[] data = Files.readAllBytes(spmFile);
        loadFromSpmBytes(data);
    }

    /**
     * Minimal SentencePiece protobuf parser.
     * The SPM model protobuf has:
     *   field 1 (pieces): repeated SentencePiece { string piece = 1; float score = 2; Type type = 3; }
     *   field 2 (trainer_spec): ...
     *   field 3 (normalizer_spec): ...
     */
    private void parseSpmProtobuf(byte[] data) {
        int pos = 0;
        int tokenId = 0;

        while (pos < data.length) {
            int tag = readProtobufVarint(data, pos);
            pos += varintSize(tag);
            int fieldNumber = tag >>> 3;
            int wireType = tag & 0x7;

            if (fieldNumber == 1 && wireType == 2) {
                // Length-delimited: a SentencePiece message
                int msgLen = readProtobufVarint(data, pos);
                pos += varintSize(msgLen);

                if (pos + msgLen > data.length) break;

                String piece = parseSentencePiece(data, pos, msgLen);
                if (piece != null) {
                    vocab.put(piece, tokenId);
                    reverseVocab.put(tokenId, piece);
                }
                tokenId++;
                pos += msgLen;
            } else if (wireType == 2) {
                // Skip length-delimited field
                int len = readProtobufVarint(data, pos);
                pos += varintSize(len) + len;
            } else if (wireType == 0) {
                // Skip varint
                readProtobufVarint(data, pos);
                pos += varintSize(readProtobufVarint(data, pos));
            } else if (wireType == 5) {
                pos += 4; // fixed32
            } else if (wireType == 1) {
                pos += 8; // fixed64
            } else {
                break; // Unknown wire type
            }
        }

        vocabSize = tokenId;

        // Register special tokens
        specialTokens.put("<pad>", PAD_TOKEN_ID);
        specialTokens.put("<eos>", EOS_TOKEN_ID);
        specialTokens.put("<bos>", BOS_TOKEN_ID);
        specialTokens.put("<unk>", UNK_TOKEN_ID);
    }

    private String parseSentencePiece(byte[] data, int offset, int length) {
        int pos = offset;
        int end = offset + length;

        while (pos < end) {
            int tag = readProtobufVarint(data, pos);
            pos += varintSize(tag);
            int fieldNumber = tag >>> 3;
            int wireType = tag & 0x7;

            if (fieldNumber == 1 && wireType == 2) {
                // piece string
                int strLen = readProtobufVarint(data, pos);
                pos += varintSize(strLen);
                if (pos + strLen <= end) {
                    return new String(data, pos, strLen, StandardCharsets.UTF_8);
                }
                pos += strLen;
            } else if (wireType == 2) {
                int len = readProtobufVarint(data, pos);
                pos += varintSize(len) + len;
            } else if (wireType == 0) {
                readProtobufVarint(data, pos);
                pos += varintSize(readProtobufVarint(data, pos));
            } else if (wireType == 5) {
                pos += 4;
            } else if (wireType == 1) {
                pos += 8;
            } else {
                break;
            }
        }
        return null;
    }

    private static int readProtobufVarint(byte[] data, int offset) {
        int result = 0;
        int shift = 0;
        int pos = offset;
        while (pos < data.length) {
            byte b = data[pos++];
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
            if (shift > 35) break; // Protect against malformed data
        }
        return result;
    }

    private static int varintSize(int value) {
        if (value < 0) return 10; // Negative values use 10 bytes in protobuf
        int size = 1;
        while (value > 0x7F) {
            size++;
            value >>>= 7;
        }
        return size;
    }

    // ===== Internal: HuggingFace tokenizer.json =====

    private void loadFromTokenizerJson(Path tokenizerJson) throws IOException {
        String content = Files.readString(tokenizerJson);
        parseTokenizerJson(content);
        type = TokenizerType.HF_JSON;
        initialized = true;
        log.info("Loaded HuggingFace tokenizer, vocab size: {}", vocabSize);
    }

    private void parseTokenizerJson(String json) {
        // Minimal JSON parser for the vocab section
        // HuggingFace tokenizer.json has: { "model": { "vocab": { "token": id, ... } } }
        // We use Jackson if available, otherwise fall back to simple parsing
        try {
            var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            var root = mapper.readTree(json);
            var model = root.get("model");
            if (model != null) {
                var vocabNode = model.get("vocab");
                if (vocabNode != null && vocabNode.isObject()) {
                        var fields = vocabNode.fields();
                        while (fields.hasNext()) {
                            Map.Entry<String, com.fasterxml.jackson.databind.JsonNode> entry = fields.next();
                            String token = entry.getKey();
                            int id = entry.getValue().asInt();
                            vocab.put(token, id);
                            reverseVocab.put(id, token);
                        }
                    vocabSize = vocab.size();
                }
            }

            // Parse added_tokens
            var addedTokens = root.get("added_tokens");
            if (addedTokens != null && addedTokens.isArray()) {
                for (var tokenNode : addedTokens) {
                    String content = tokenNode.has("content") ? tokenNode.get("content").asText() : null;
                    int id = tokenNode.has("id") ? tokenNode.get("id").asInt() : -1;
                    if (content != null && id >= 0) {
                        specialTokens.put(content, id);
                        vocab.put(content, id);
                        reverseVocab.put(id, content);
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse tokenizer.json", e);
        }
    }

    // ===== Internal: Vocabulary-based encoding =====

    private int[] encodeWithVocab(String text) {
        // Check special tokens first
        for (var entry : specialTokens.entrySet()) {
            if (text.equals(entry.getKey())) {
                return new int[] { entry.getValue() };
            }
        }

        // Greedy longest-match encoding
        List<Integer> ids = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            int bestLen = 0;
            int bestId = UNK_TOKEN_ID;

            // Try matching from longest to shortest
            for (int len = Math.min(text.length() - i, 32); len >= 1; len--) {
                String sub = text.substring(i, i + len);
                // Try with SentencePiece's space prefix
                String withPrefix = "▁" + sub;
                Integer id = vocab.get(sub);
                if (id == null && i == 0 || (i > 0 && text.charAt(i - 1) == ' ')) {
                    id = vocab.get(withPrefix);
                }
                if (id == null) {
                    id = vocab.get(sub);
                }
                if (id != null) {
                    bestLen = len;
                    bestId = id;
                    break;
                }
            }

            if (bestLen == 0) {
                // Character not in vocab, use byte fallback
                byte[] bytes = text.substring(i, i + 1).getBytes(StandardCharsets.UTF_8);
                for (byte b : bytes) {
                    // Byte tokens in SPM are <0xHH> format
                    String byteToken = String.format("<0x%02X>", b & 0xFF);
                    Integer id = vocab.get(byteToken);
                    ids.add(id != null ? id : UNK_TOKEN_ID);
                }
                i++;
            } else {
                ids.add(bestId);
                i += bestLen;
            }
        }

        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    private String decodeWithVocab(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (id == BOS_TOKEN_ID || id == PAD_TOKEN_ID) continue;
            if (id == EOS_TOKEN_ID) break;

            String piece = reverseVocab.get(id);
            if (piece != null) {
                // SentencePiece uses ▁ (U+2581) for space
                sb.append(piece.replace("▁", " "));
            }
        }
        // Trim leading space if present
        if (!sb.isEmpty() && sb.charAt(0) == ' ') {
            sb.deleteCharAt(0);
        }
        return sb.toString();
    }

    // ===== Internal: Byte-level fallback =====

    private void initByteFallback() {
        vocab.clear();
        reverseVocab.clear();
        // Map first 256 entries to bytes
        for (int i = 0; i < 256; i++) {
            String s = String.valueOf((char) i);
            vocab.put(s, i + 4); // Offset by 4 for special tokens
            reverseVocab.put(i + 4, s);
        }
        vocabSize = 260;
        type = TokenizerType.BYTE_FALLBACK;
        initialized = true;
    }

    private int[] encodeByteFallback(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = (bytes[i] & 0xFF) + 4; // Offset by 4 for special tokens
        }
        return ids;
    }

    private String decodeByteFallback(int[] tokenIds) {
        byte[] bytes = new byte[tokenIds.length];
        int count = 0;
        for (int id : tokenIds) {
            if (id >= 4 && id < 260) {
                bytes[count++] = (byte) (id - 4);
            }
        }
        return new String(bytes, 0, count, StandardCharsets.UTF_8);
    }

    // ===== Internal: Safetensor tokenizer fallback =====

    /**
     * Search for tokenizer.json in the safetensors variant of the model.
     * When a model is downloaded as LiteRT, the original safetensor variant
     * (with tokenizer.json) may already exist at a parallel path.
     *
     * E.g., LiteRT model: ~/.gollek/models/litert/litert-community/gemma-4-E2B-it-litert-lm/
     *       Safetensor:    ~/.gollek/models/safetensors/google/gemma-4-E2B-it/
     */
    private static Optional<Path> findCompanionTokenizerJson(Path modelPath) {
        try {
            // Extract model name hint from the path
            String fileName = modelPath.getFileName().toString();
            Path parentDir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
            String dirName = parentDir != null ? parentDir.getFileName().toString() : fileName;
            String fileHint = fileName.toLowerCase()
                    .replaceAll("(_qualcomm_[^.]+)?\\.(litertlm|task|tflite|tfl)$", "")
                    .replace("-web", "")
                    .replace("-litert-lm", "")
                    .replace("-litert", "");
            String dirHint = dirName.toLowerCase()
                    .replace("-litert-lm", "")
                    .replace("-litert", "")
                    .replace("litert-community/", "");
            List<String> searchHints = new ArrayList<>();
            if (!fileHint.isBlank()) {
                searchHints.add(fileHint);
            }
            if (!dirHint.isBlank() && !dirHint.equals(fileHint)) {
                searchHints.add(dirHint);
            }

            List<Path> roots = Arrays.asList(
                    Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs"),
                    Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors"));
            for (Path modelsRoot : roots) {
                if (!Files.isDirectory(modelsRoot)) {
                    continue;
                }
                try (var stream = Files.walk(modelsRoot, 4)) {
                    Optional<Path> tokenizer = stream
                            .filter(Files::isRegularFile)
                            .filter(p -> p.getFileName().toString().equals("tokenizer.json"))
                            .filter(p -> {
                                String pathLower = p.toString().toLowerCase();
                                for (String hint : searchHints) {
                                    if (!hint.isBlank() && pathLower.contains(hint)) {
                                        return true;
                                    }
                                }
                                return false;
                            })
                            .findFirst();
                    if (tokenizer.isPresent()) {
                        log.info("Found companion tokenizer.json at {}", tokenizer.get());
                        return tokenizer;
                    }
                }
            }
        } catch (Exception e) {
            log.debug("Failed to search for companion tokenizer.json: {}", e.getMessage());
        }
        return Optional.empty();
    }

    private static int[] toIntArray(long[] values) {
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (int) values[i];
        }
        return out;
    }

    private static long[] toLongArray(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }
}
