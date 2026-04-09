package tech.kayys.gollek.ml.data.multimodal;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.Dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A dataset that pairs images with their corresponding text descriptions.
 *
 * <p>This dataset implementation loads image-text pairs from a directory structure where each image
 * has a corresponding text file with the same basename. For example, {@code image.jpg} is paired
 * with {@code image.txt}. This is useful for vision-language tasks such as image captioning,
 * visual question answering, and image-text retrieval models.
 *
 * <p><b>Supported Image Formats:</b> JPG, JPEG, PNG (case-insensitive extensions)
 *
 * <p><b>Expected Directory Structure:</b>
 * <pre>
 * dataset_directory/
 * ├── photo1.jpg
 * ├── photo1.txt          # paired description
 * ├── photo2.png
 * ├── photo2.txt          # paired description
 * ├── photo3.jpeg
 * ├── photo3.txt          # paired description
 * ├── subdir/
 * │   ├── image4.jpg
 * │   └── image4.txt      # paired description
 * └── ...
 *
 * # Files without pairs are automatically skipped:
 * ├── unpaired.jpg        # skipped (no .txt file)
 * └── orphan.txt          # ignored (no matching image)
 * </pre>
 *
 * <p><b>Text File Format:</b>
 * <ul>
 *   <li>Plain text format (UTF-8 encoding)</li>
 *   <li>May be single-line or multi-line</li>
 *   <li>Leading/trailing whitespace is automatically trimmed</li>
 *   <li>Examples: descriptions, captions, labels, or any text metadata</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>
 * Path datasetDir = Paths.get("data/image_text_pairs");
 * ImageTextDataset dataset = new ImageTextDataset(datasetDir);
 * 
 * // Get a paired sample
 * ImageTextDataset.Sample sample = dataset.get(0);
 * GradTensor imageData = sample.image();
 * String textDescription = sample.text();
 * 
 * // Useful for training models:
 * for (int i = 0; i &lt; dataset.size(); i++) {
 *     ImageTextDataset.Sample sample = dataset.get(i);
 *     // Feed image tensor and text to your vision-language model
 *     model.train(sample.image(), sample.text());
 * }
 * </pre>
 *
 * <p><b>Performance Characteristics:</b>
 * <ul>
 *   <li><b>Construction Time:</b> O(n) where n is total files in directory and subdirectories.
 *                                  Performs directory walk and file pairing validation.</li>
 *   <li><b>Memory Usage:</b> O(m) for storing image and text file paths, where m is number of
 *                           valid image-text pairs. Invalid/unpaired files are ignored.</li>
 *   <li><b>Access Time:</b> O(1) lookup by index; O(k) for loading image and text data
 *                          where k is combined size of files.</li>
 * </ul>
 *
 * <p><b>Pairing Logic:</b>
 * <ul>
 *   <li>For each image file found, the dataset looks for a text file with the same base name</li>
 *   <li>Example: {@code photo.jpg} is paired with {@code photo.txt}</li>
 *   <li>Image files without corresponding text files are silently skipped</li>
 *   <li>Text files without corresponding images are ignored</li>
 * </ul>
 *
 * <p><b>Error Handling:</b> If a paired file fails to load during {@link #get(int)}, a
 * {@link RuntimeException} is thrown containing the sample index and original cause.
 *
 * <p><b>Thread Safety:</b> This class is thread-safe for read access. Multiple threads can safely
 * call {@link #get(int)} concurrently.
 *
 * @see Sample
 * @see GradTensor
 * @see Dataset
 */
public class ImageTextDataset implements Dataset<ImageTextDataset.Sample> {

    /**
     * A record representing a paired image-text sample.
     *
     * <p>This record encapsulates the image tensor and corresponding text description from a
     * single paired sample in the dataset.
     *
     * @param image a {@link GradTensor} representing the image in [C, H, W] format
     *              (channels, height, width). Commonly [3, 224, 224] for RGB or similar dimensions.
     * @param text  the textual description or annotation for the image. Non-null, whitespace-trimmed.
     */
    public record Sample(GradTensor image, String text) {}

    private final List<Path> imagePaths = new ArrayList<>();
    private final List<Path> textPaths = new ArrayList<>();

    /**
     * Constructs an image-text dataset that scans the specified directory for paired files.
     *
     * <p>This constructor recursively walks the directory tree, collecting all image files
     * (.jpg, .jpeg, .png) and attempting to pair them with corresponding text files (.txt)
     * having the same base name. Only successfully paired files are included in the dataset.
     * Unpaired images and orphan text files are silently ignored.
     *
     * <p>Image and text paths are cached during construction for efficient subsequent access.
     * Actual image and text data is not loaded until {@link #get(int)} is called.
     *
     * @param directory the path to the directory containing image-text pairs.
     *                  The directory is recursively scanned for paired files.
     *                  Must not be null and must be a valid directory.
     * @throws IOException if an I/O error occurs while reading the directory structure
     * @throws NullPointerException if {@code directory} is null
     * @see #get(int) for loading actual image-text pairs
     */
    public ImageTextDataset(Path directory) throws IOException {
        try (var stream = Files.walk(directory)) {
            stream.filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.toString().toLowerCase();
                    return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
                })
                .forEach(imgPath -> {
                    String imgName = imgPath.getFileName().toString();
                    String textName = imgName.substring(0, imgName.lastIndexOf('.')) + ".txt";
                    Path textPath = imgPath.getParent().resolve(textName);
                    if (Files.exists(textPath)) {
                        imagePaths.add(imgPath);
                        textPaths.add(textPath);
                    }
                });
        }
    }

    /**
     * Retrieves an image-text pair at the specified index.
     *
     * <p>This method loads both the image and text files. The image is converted to a
     * {@link GradTensor} in channel-height-width format [C, H, W], and the text is read as a
     * UTF-8 encoded string with leading/trailing whitespace trimmed. Both are packaged in a
     * {@link Sample} record.
     *
     * <p>Lazy loading is used: image and text data are only read when explicitly requested,
     * not during dataset construction.
     *
     * @param index the zero-based index of the pair to retrieve
     * @return a {@link Sample} containing the image tensor and text description
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     *         ({@code index < 0 || index >= size()})
     * @throws RuntimeException if either the image or text file cannot be read/loaded,
     *         with the original {@link IOException} as cause. The exception message includes
     *         the sample index for context.
     * @see Sample
     */
    @Override
    public Sample get(int index) {
        Path imgPath = imagePaths.get(index);
        Path txtPath = textPaths.get(index);
        try {
            GradTensor image = GradTensor.fromImage(javax.imageio.ImageIO.read(imgPath.toFile()));
            String text = Files.readString(txtPath).trim();
            return new Sample(image, text);
        } catch (IOException e) {
            throw new RuntimeException("Error loading sample at index " + index, e);
        }
    }

    /**
     * Returns the total number of image-text pairs in this dataset.
     *
     * <p>This is the count of successfully paired images and text files found during
     * dataset construction. Unpaired files are not included in this count.
     *
     * @return the number of valid image-text pairs in the dataset
     */
    @Override
    public int size() {
        return imagePaths.size();
    }
}
