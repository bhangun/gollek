package tech.kayys.gollek.sdk.feature;

import tech.kayys.gollek.sdk.exception.SdkException;

import java.util.List;
import java.util.Map;

/**
 * ML Kit-like feature surface for application SDK users.
 *
 * <p>This API groups high-level on-device or local-runtime capabilities:
 * face recognition, natural language, and document scanning.</p>
 */
public interface GollekFeatureKit {

    FaceRecognitionService faceRecognition();

    NaturalLanguageService naturalLanguage();

    DocumentScannerService documentScanner();

    interface FaceRecognitionService {
        FaceDetectionResult detectFaces(FaceDetectionRequest request) throws SdkException;

        FaceComparisonResult compareFaces(FaceComparisonRequest request) throws SdkException;
    }

    interface NaturalLanguageService {
        LanguageDetectionResult detectLanguage(LanguageDetectionRequest request) throws SdkException;

        SentimentAnalysisResult analyzeSentiment(SentimentRequest request) throws SdkException;

        EntityExtractionResult extractEntities(EntityExtractionRequest request) throws SdkException;
    }

    interface DocumentScannerService {
        DocumentScanResult scan(DocumentScanRequest request) throws SdkException;
    }

    enum SentimentLabel {
        POSITIVE,
        NEGATIVE,
        NEUTRAL,
        MIXED
    }

    record ImageInput(byte[] bytes, String uri, String mimeType) {
        public ImageInput {
            if ((bytes == null || bytes.length == 0) && (uri == null || uri.isBlank())) {
                throw new IllegalArgumentException("ImageInput requires either bytes or uri.");
            }
            if (bytes != null && bytes.length > 0 && uri != null && !uri.isBlank()) {
                throw new IllegalArgumentException("ImageInput accepts bytes or uri, not both.");
            }
            bytes = bytes == null ? null : bytes.clone();
            mimeType = mimeType == null || mimeType.isBlank() ? "application/octet-stream" : mimeType;
        }

        public static ImageInput fromBytes(byte[] bytes, String mimeType) {
            return new ImageInput(bytes, null, mimeType);
        }

        public static ImageInput fromUri(String uri, String mimeType) {
            return new ImageInput(null, uri, mimeType);
        }

        public boolean hasInlineBytes() {
            return bytes != null && bytes.length > 0;
        }
    }

    record BoundingBox(double x, double y, double width, double height) {
    }

    record DetectedFace(String faceId, double confidence, BoundingBox box) {
        public DetectedFace {
            faceId = faceId == null || faceId.isBlank() ? "face" : faceId;
        }
    }

    record FaceDetectionRequest(String model, ImageInput image, double minConfidence) {
        public FaceDetectionRequest {
            if (image == null) {
                throw new IllegalArgumentException("FaceDetectionRequest.image is required.");
            }
            minConfidence = normalizeProbability(minConfidence, 0.5d);
        }
    }

    record FaceDetectionResult(
            String model,
            List<DetectedFace> faces,
            String rawOutput,
            Map<String, Object> metadata) {
        public FaceDetectionResult {
            faces = faces == null ? List.of() : List.copyOf(faces);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record FaceComparisonRequest(String model, ImageInput firstImage, ImageInput secondImage, double threshold) {
        public FaceComparisonRequest {
            if (firstImage == null || secondImage == null) {
                throw new IllegalArgumentException("FaceComparisonRequest requires both firstImage and secondImage.");
            }
            threshold = normalizeProbability(threshold, 0.75d);
        }
    }

    record FaceComparisonResult(
            String model,
            boolean match,
            double similarity,
            double threshold,
            String rawOutput,
            Map<String, Object> metadata) {
        public FaceComparisonResult {
            similarity = normalizeProbability(similarity, 0.0d);
            threshold = normalizeProbability(threshold, 0.75d);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record LanguageDetectionRequest(String model, String text) {
        public LanguageDetectionRequest {
            if (text == null || text.isBlank()) {
                throw new IllegalArgumentException("LanguageDetectionRequest.text is required.");
            }
        }
    }

    record LanguageDetectionResult(
            String model,
            String languageCode,
            double confidence,
            String rawOutput,
            Map<String, Object> metadata) {
        public LanguageDetectionResult {
            languageCode = languageCode == null || languageCode.isBlank() ? "und" : languageCode;
            confidence = normalizeProbability(confidence, 0.0d);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record SentimentRequest(String model, String text) {
        public SentimentRequest {
            if (text == null || text.isBlank()) {
                throw new IllegalArgumentException("SentimentRequest.text is required.");
            }
        }
    }

    record SentimentAnalysisResult(
            String model,
            SentimentLabel label,
            double positiveScore,
            double neutralScore,
            double negativeScore,
            String rawOutput,
            Map<String, Object> metadata) {
        public SentimentAnalysisResult {
            label = label == null ? SentimentLabel.NEUTRAL : label;
            positiveScore = normalizeProbability(positiveScore, 0.0d);
            neutralScore = normalizeProbability(neutralScore, 0.0d);
            negativeScore = normalizeProbability(negativeScore, 0.0d);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record Entity(String text, String type, int start, int end, double confidence) {
        public Entity {
            text = text == null ? "" : text;
            type = type == null || type.isBlank() ? "unknown" : type;
            confidence = normalizeProbability(confidence, 0.0d);
        }
    }

    record EntityExtractionRequest(String model, String text) {
        public EntityExtractionRequest {
            if (text == null || text.isBlank()) {
                throw new IllegalArgumentException("EntityExtractionRequest.text is required.");
            }
        }
    }

    record EntityExtractionResult(String model, List<Entity> entities, String rawOutput, Map<String, Object> metadata) {
        public EntityExtractionResult {
            entities = entities == null ? List.of() : List.copyOf(entities);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    record DocumentScanRequest(
            String model,
            ImageInput document,
            boolean detectLayout,
            boolean normalizeOrientation,
            String outputFormat) {
        public DocumentScanRequest {
            if (document == null) {
                throw new IllegalArgumentException("DocumentScanRequest.document is required.");
            }
            outputFormat = outputFormat == null || outputFormat.isBlank() ? "text" : outputFormat;
        }
    }

    record ScannedBlock(int page, String type, String text, BoundingBox box, double confidence) {
        public ScannedBlock {
            type = type == null || type.isBlank() ? "text" : type;
            text = text == null ? "" : text;
            confidence = normalizeProbability(confidence, 0.0d);
        }
    }

    record DocumentScanResult(
            String model,
            String text,
            List<ScannedBlock> blocks,
            String rawOutput,
            Map<String, Object> metadata) {
        public DocumentScanResult {
            text = text == null ? "" : text;
            blocks = blocks == null ? List.of() : List.copyOf(blocks);
            rawOutput = rawOutput == null ? "" : rawOutput;
            metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        }
    }

    private static double normalizeProbability(double value, double fallback) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return fallback;
        }
        if (value < 0.0d) {
            return 0.0d;
        }
        if (value > 1.0d) {
            return 1.0d;
        }
        return value;
    }
}
