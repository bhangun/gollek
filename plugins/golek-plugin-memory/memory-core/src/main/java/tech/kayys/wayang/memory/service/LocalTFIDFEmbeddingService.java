package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Local embedding service using simple TF-IDF approach
 * For development and testing when external APIs are not available
 */
@ApplicationScoped
public class LocalTFIDFEmbeddingService implements EmbeddingService {

    private static final Logger LOG = LoggerFactory.getLogger(LocalTFIDFEmbeddingService.class);
    private static final int DIMENSION = 384; // Common for local models
    private static final Pattern WORD_PATTERN = Pattern.compile("\\w+");

    // Document frequency: word -> count
    private final Map<String, Integer> documentFrequency = new ConcurrentHashMap<>();
    private final Set<String> vocabulary = ConcurrentHashMap.newKeySet();
    private int totalDocuments = 0;

    @Override
    public Uni<float[]> embed(String text) {
        LOG.debug("Generating local TF-IDF embedding for: {} chars", text.length());

        return Uni.createFrom().item(() -> {
            // Tokenize and calculate term frequency
            Map<String, Integer> termFrequency = calculateTermFrequency(text);

            // Update global statistics
            updateDocumentStatistics(termFrequency.keySet());

            // Generate TF-IDF vector
            float[] embedding = generateTFIDFVector(termFrequency);

            // Normalize
            normalize(embedding);

            return embedding;
        });
    }

    @Override
    public Uni<List<float[]>> embedBatch(List<String> texts) {
        LOG.debug("Generating local embeddings for batch of {} texts", texts.size());

        return Uni.createFrom().item(() -> {
            List<float[]> embeddings = new ArrayList<>();

            for (String text : texts) {
                Map<String, Integer> termFrequency = calculateTermFrequency(text);
                updateDocumentStatistics(termFrequency.keySet());
                float[] embedding = generateTFIDFVector(termFrequency);
                normalize(embedding);
                embeddings.add(embedding);
            }

            return embeddings;
        });
    }

    @Override
    public int getDimension() {
        return DIMENSION;
    }

    @Override
    public String getProvider() {
        return "local-tfidf";
    }

    /**
     * Calculate term frequency for text
     */
    private Map<String, Integer> calculateTermFrequency(String text) {
        Map<String, Integer> tf = new HashMap<>();

        var matcher = WORD_PATTERN.matcher(text.toLowerCase());
        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2) { // Skip short words
                tf.merge(word, 1, Integer::sum);
            }
        }

        return tf;
    }

    /**
     * Update global document statistics
     */
    private synchronized void updateDocumentStatistics(Set<String> words) {
        totalDocuments++;

        for (String word : words) {
            vocabulary.add(word);
            documentFrequency.merge(word, 1, Integer::sum);
        }
    }

    /**
     * Generate TF-IDF vector
     */
    private float[] generateTFIDFVector(Map<String, Integer> termFrequency) {
        float[] vector = new float[DIMENSION];

        // Convert vocabulary to indexed list for consistent hashing
        List<String> vocabList = new ArrayList<>(vocabulary);
        Collections.sort(vocabList);

        for (Map.Entry<String, Integer> entry : termFrequency.entrySet()) {
            String term = entry.getKey();
            int tf = entry.getValue();

            // Calculate IDF
            int df = documentFrequency.getOrDefault(term, 1);
            double idf = Math.log((double) (totalDocuments + 1) / (df + 1));

            // Calculate TF-IDF
            double tfidf = tf * idf;

            // Hash term to dimension
            int index = Math.abs(term.hashCode()) % DIMENSION;
            vector[index] += (float) tfidf;
        }

        return vector;
    }

    /**
     * Normalize vector to unit length
     */
    private void normalize(float[] vector) {
        double sum = 0.0;
        for (float v : vector) {
            sum += v * v;
        }

        if (sum > 0) {
            double norm = Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}