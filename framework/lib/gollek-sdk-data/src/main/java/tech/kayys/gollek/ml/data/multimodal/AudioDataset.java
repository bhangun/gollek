package tech.kayys.gollek.ml.data.multimodal;

import tech.kayys.gollek.ml.data.Dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A dataset that loads audio files from a directory and provides raw byte access.
 *
 * <p>This dataset implementation scans a directory for audio files (supporting WAV, MP3, and FLAC formats)
 * and loads them as raw byte arrays. Audio data is loaded lazily on demand when {@link #get(int)} is called,
 * enabling efficient memory usage for large audio collections. The raw bytes preserve the original audio
 * encoding and can be further processed or decoded as needed.
 *
 * <p><b>Supported Formats:</b> WAV, MP3, FLAC (case-insensitive extensions)
 *
 * <p><b>Directory Structure:</b>
 * <pre>
 * audio_directory/
 * ├── audio1.wav
 * ├── audio2.mp3
 * ├── subdir/
 * │   ├── audio3.flac
 * │   └── audio4.wav
 * └── ...
 * </pre>
 *
 * <p><b>Return Format:</b>
 * <ul>
 *   <li><b>Raw Bytes:</b> Complete audio file contents as-is, preserving original encoding</li>
 *   <li><b>Size:</b> Array length equals the file size in bytes</li>
 *   <li><b>Decoding:</b> Further processing (e.g., PCM conversion) handled by client code</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>
 * Path audioDir = Paths.get("data/audio");
 * AudioDataset dataset = new AudioDataset(audioDir);
 * 
 * // Get first audio file as byte array
 * byte[] audioBytes = dataset.get(0);
 * 
 * // Get the original file path for metadata or decoding with external library
 * Path audioPath = dataset.getPath(0);
 * 
 * // Use with audio processing library (e.g., for WAV/MP3 decoding)
 * AudioInputStream stream = AudioSystem.getAudioInputStream(audioPath.toFile());
 * </pre>
 *
 * <p><b>Performance Characteristics:</b>
 * <ul>
 *   <li><b>Construction Time:</b> O(n) where n is total files in directory and subdirectories
 *                                  (performs single directory walk to collect audio file paths)</li>
 *   <li><b>Memory Usage:</b> O(n) for storing audio file paths; actual audio data loaded on demand</li>
 *   <li><b>Access Time:</b> O(1) lookup by index; O(k) audio loading where k is file size</li>
 * </ul>
 *
 * <p><b>Error Handling:</b> If an audio file fails to load during {@link #get(int)}, a
 * {@link RuntimeException} is thrown containing the file path and original cause.
 *
 * <p><b>Thread Safety:</b> This class is thread-safe for read access. Multiple threads can safely
 * call {@link #get(int)} and {@link #getPath(int)} concurrently.
 *
 * @see Dataset
 */
public class AudioDataset implements Dataset<byte[]> {

    private final List<Path> audioPaths;

    /**
     * Constructs an audio dataset that scans the specified directory for audio files.
     *
     * <p>This constructor recursively walks the directory tree, collecting all files with
     * supported audio extensions (.wav, .mp3, .flac). The collected paths are cached for
     * efficient subsequent access. Actual audio data is not loaded until {@link #get(int)} is called.
     *
     * @param directory the path to the directory containing audio files.
     *                  The directory is recursively scanned for audio files.
     *                  Must not be null and must be a valid directory.
     * @throws IOException if an I/O error occurs while reading the directory structure
     * @throws NullPointerException if {@code directory} is null
     */
    public AudioDataset(Path directory) throws IOException {
        try (var stream = Files.walk(directory)) {
            this.audioPaths = stream
                .filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.toString().toLowerCase();
                    return name.endsWith(".wav") || name.endsWith(".mp3") || name.endsWith(".flac");
                })
                .collect(Collectors.toList());
        }
    }

    /**
     * Retrieves an audio file at the specified index as raw bytes.
     *
     * <p>The entire audio file is read into memory as a byte array. The bytes are in their
     * original encoding format (WAV, MP3, or FLAC) and may require decoding for further processing.
     * This method performs lazy loading, meaning the audio data is only read when explicitly requested.
     *
     * @param index the zero-based index of the audio file to retrieve
     * @return a byte array containing the complete audio file data
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     *         ({@code index < 0 || index >= size()})
     * @throws RuntimeException if the audio file cannot be read, with the original
     *         {@link IOException} as cause
     */
    @Override
    public byte[] get(int index) {
        Path path = audioPaths.get(index);
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load audio: " + path, e);
        }
    }

    /**
     * Returns the total number of audio files in this dataset.
     *
     * @return the number of audio files found in the directory and subdirectories
     */
    @Override
    public int size() {
        return audioPaths.size();
    }

    /**
     * Retrieves the file path of the audio file at the specified index.
     *
     * <p>This method is useful for obtaining metadata about the audio source, determining its
     * format, or for logging and debugging purposes.
     *
     * @param index the zero-based index of the audio file
     * @return the {@link Path} to the audio file
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     *         ({@code index < 0 || index >= size()})
     */
    public Path getPath(int index) {
        return audioPaths.get(index);
    }
}
