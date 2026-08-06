package tech.kayys.gollek.onnx.runner;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class MossSentencePieceTokenizer {
    private static final String SPACE = "\u2581";
    private static final double UNKNOWN_PENALTY = -100.0;

    private final Map<String, Piece> piecesByText = new HashMap<>();
    private int maxPieceLength = 1;
    private int unknownTokenId = 0;

    static MossSentencePieceTokenizer load(Path modelPath) throws IOException {
        MossSentencePieceTokenizer tokenizer = new MossSentencePieceTokenizer();
        tokenizer.parse(Files.readAllBytes(modelPath));
        if (tokenizer.piecesByText.isEmpty()) {
            throw new IOException("No SentencePiece pieces found in " + modelPath);
        }
        return tokenizer;
    }

    int[] encode(String text) {
        String normalized = normalize(text);
        int length = normalized.length();
        double[] bestScore = new double[length + 1];
        int[] bestTokenId = new int[length + 1];
        int[] bestTokenLength = new int[length + 1];

        for (int i = 0; i < length; i++) {
            bestScore[i] = Double.NEGATIVE_INFINITY;
            bestTokenId[i] = unknownTokenId;
            bestTokenLength[i] = 1;
        }
        bestScore[length] = 0.0;

        for (int i = length - 1; i >= 0; i--) {
            int maxLen = Math.min(maxPieceLength, length - i);
            for (int pieceLength = 1; pieceLength <= maxLen; pieceLength++) {
                Piece piece = piecesByText.get(normalized.substring(i, i + pieceLength));
                if (piece == null || Double.isInfinite(bestScore[i + pieceLength])) {
                    continue;
                }
                double score = piece.score + bestScore[i + pieceLength];
                if (score > bestScore[i]) {
                    bestScore[i] = score;
                    bestTokenId[i] = piece.id;
                    bestTokenLength[i] = pieceLength;
                }
            }
            if (Double.isInfinite(bestScore[i])) {
                int fallbackLength = Character.charCount(normalized.codePointAt(i));
                bestScore[i] = UNKNOWN_PENALTY + bestScore[Math.min(length, i + fallbackLength)];
                bestTokenId[i] = unknownTokenId;
                bestTokenLength[i] = fallbackLength;
            }
        }

        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < length;) {
            ids.add(bestTokenId[i]);
            i += Math.max(1, bestTokenLength[i]);
        }
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    private static String normalize(String text) {
        String value = text == null ? "" : text.strip();
        value = value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ');
        while (value.contains("  ")) {
            value = value.replace("  ", " ");
        }
        if (value.isEmpty()) {
            return SPACE;
        }
        return SPACE + value.replace(" ", SPACE);
    }

    private void parse(byte[] data) {
        int pos = 0;
        int tokenId = 0;
        while (pos < data.length) {
            Varint tag = readVarint(data, pos);
            if (!tag.valid()) {
                break;
            }
            pos = tag.nextOffset();
            int fieldNumber = tag.value() >>> 3;
            int wireType = tag.value() & 0x7;

            if (fieldNumber == 1 && wireType == 2) {
                Varint length = readVarint(data, pos);
                if (!length.valid()) {
                    break;
                }
                pos = length.nextOffset();
                int end = pos + length.value();
                if (end > data.length) {
                    break;
                }
                Piece piece = parsePiece(data, pos, length.value(), tokenId);
                if (piece != null && piece.text != null && !piece.text.isEmpty()) {
                    piecesByText.put(piece.text, piece);
                    maxPieceLength = Math.max(maxPieceLength, piece.text.length());
                    if ("<unk>".equals(piece.text) || "\u2047".equals(piece.text)) {
                        unknownTokenId = piece.id;
                    }
                }
                tokenId++;
                pos = end;
            } else {
                pos = skipField(data, pos, wireType);
                if (pos < 0) {
                    break;
                }
            }
        }
    }

    private Piece parsePiece(byte[] data, int offset, int length, int tokenId) {
        int pos = offset;
        int end = offset + length;
        String text = null;
        float score = 0.0f;
        int type = 1;

        while (pos < end) {
            Varint tag = readVarint(data, pos);
            if (!tag.valid()) {
                break;
            }
            pos = tag.nextOffset();
            int fieldNumber = tag.value() >>> 3;
            int wireType = tag.value() & 0x7;

            if (fieldNumber == 1 && wireType == 2) {
                Varint strLen = readVarint(data, pos);
                if (!strLen.valid()) {
                    break;
                }
                pos = strLen.nextOffset();
                if (pos + strLen.value() <= end) {
                    text = new String(data, pos, strLen.value(), StandardCharsets.UTF_8);
                }
                pos += strLen.value();
            } else if (fieldNumber == 2 && wireType == 5) {
                if (pos + 4 <= end) {
                    score = ByteBuffer.wrap(data, pos, 4).order(ByteOrder.LITTLE_ENDIAN).getFloat();
                }
                pos += 4;
            } else if (fieldNumber == 3 && wireType == 0) {
                Varint typeValue = readVarint(data, pos);
                if (!typeValue.valid()) {
                    break;
                }
                type = typeValue.value();
                pos = typeValue.nextOffset();
            } else {
                int skipped = skipField(data, pos, wireType);
                if (skipped < 0 || skipped > end) {
                    break;
                }
                pos = skipped;
            }
        }
        return text == null ? null : new Piece(tokenId, text, score, type);
    }

    private static int skipField(byte[] data, int pos, int wireType) {
        return switch (wireType) {
            case 0 -> {
                Varint value = readVarint(data, pos);
                yield value.valid() ? value.nextOffset() : -1;
            }
            case 1 -> pos + 8 <= data.length ? pos + 8 : -1;
            case 2 -> {
                Varint length = readVarint(data, pos);
                yield length.valid() && length.nextOffset() + length.value() <= data.length
                        ? length.nextOffset() + length.value()
                        : -1;
            }
            case 5 -> pos + 4 <= data.length ? pos + 4 : -1;
            default -> -1;
        };
    }

    private static Varint readVarint(byte[] data, int offset) {
        int result = 0;
        int shift = 0;
        int pos = offset;
        while (pos < data.length && shift < 35) {
            int b = data[pos++] & 0xff;
            result |= (b & 0x7f) << shift;
            if ((b & 0x80) == 0) {
                return new Varint(result, pos, true);
            }
            shift += 7;
        }
        return new Varint(0, offset, false);
    }

    private record Piece(int id, String text, float score, int type) {
    }

    private record Varint(int value, int nextOffset, boolean valid) {
    }
}
