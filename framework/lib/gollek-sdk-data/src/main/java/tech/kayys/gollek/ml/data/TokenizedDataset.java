package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.function.UnaryOperator;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

/**
 * Text dataset — loads text samples from a directory or file for NLP training.
 *
 * <p>
 * Supports two formats:
 * <ul>
 * <li>Classification: directory with one subdirectory per class</li>
 * <li>Language modeling: single text file, split into fixed-length chunks</li>
 * </ul>
 *
 * <h3>Example — classification</h3>
 * 
 * <pre>{@code
 * // Directory structure: data/pos/*.txt, data/neg/*.txt
 * var dataset = TokenizedDataset.fromDirectory(Path.of("data"), tokenizer, maxLength = 128);
 * }</pre>
 *
 * <h3>Example — language modeling</h3>
 * 
 * <pre>{@code
 * var dataset = TokenizedDataset.fromFile(Path.of("corpus.txt"), tokenizer, chunkSize = 512);
 * }</pre>
 */
public final class TokenizedDataset implements Dataset<Dataset.Sample> {

    private final List<Sample> samples;

    private TokenizedDataset(List<Sample> samples) {
        this.samples = samples;
    }

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public Sample get(int index) {
        return samples.get(index);
    }

    /**
     * Loads a text classification dataset from a directory.
     *
     * <p>
     * Each subdirectory is treated as a class label (sorted alphabetically).
     * Each {@code .txt} file in a subdirectory is one sample.
     *
     * @param dir       root directory containing class subdirectories
     * @param tokenizer tokenizer for encoding text
     * @param maxLength maximum token sequence length
     * @return loaded dataset
     * @throws IOException if the directory cannot be read
     */
    public static TokenizedDataset fromDirectory(
            Path dir,
            Tokenizer tokenizer,
            int maxLength) throws IOException {

        List<Path> classDirs = new ArrayList<>();
        try (var stream = Files.list(dir)) {
            stream.filter(Files::isDirectory).sorted().forEach(classDirs::add);
        }

        List<Sample> samples = new ArrayList<>();
        for (int classIdx = 0; classIdx < classDirs.size(); classIdx++) {
            final int label = classIdx;
            try (var files = Files.walk(classDirs.get(classIdx), 1)) {
                files.filter(p -> p.toString().endsWith(".txt")).forEach(file -> {
                    try {
                        String text = Files.readString(file);
                        long[] tokens = tokenizer.encode(text,
                                EncodeOptions.defaultOptions());
                        float[] ids = new float[maxLength];
                        int padId = tokenizer.padTokenId();
                        for (int i = 0; i < maxLength; i++) {
                            ids[i] = i < tokens.length ? (float) tokens[i] : (float) padId;
                        }
                        samples.add(new Sample(
                                GradTensor.of(ids, maxLength),
                                GradTensor.scalar(label)));
                    } catch (IOException e) {
                        /* skip unreadable files */ }
                });
            }
        }
        return new TokenizedDataset(samples);
    }

    /**
     * Loads a language modeling dataset from a single text file.
     *
     * <p>
     * Tokenizes the entire file and splits into fixed-length chunks.
     * Input = chunk[0..n-1], label = chunk[1..n] (next-token prediction).
     *
     * @param file      path to text file
     * @param tokenizer tokenizer for encoding
     * @param chunkSize number of tokens per chunk
     * @return loaded dataset
     * @throws IOException if the file cannot be read
     */
    public static TokenizedDataset fromFile(
            Path file,
            tech.kayys.gollek.tokenizer.spi.Tokenizer tokenizer,
            int chunkSize) throws IOException {

        String text = Files.readString(file);
        long[] allIds = tokenizer.encode(text, tech.kayys.gollek.tokenizer.spi.EncodeOptions.defaultOptions());
        List<Sample> samples = new ArrayList<>();

        for (int i = 0; i + chunkSize + 1 <= allIds.length; i += chunkSize) {
            float[] input = new float[chunkSize];
            float[] target = new float[chunkSize];
            for (int j = 0; j < chunkSize; j++) {
                input[j] = (float) allIds[i + j];
                target[j] = (float) allIds[i + j + 1];
            }
            samples.add(new Sample(
                    GradTensor.of(input, chunkSize),
                    GradTensor.of(target, chunkSize)));
        }
        return new TokenizedDataset(samples);
    }
}
