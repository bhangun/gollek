package tech.kayys.gollek.ml.data.multimodal;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.data.Dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A dataset that loads images from a directory and converts them to tensors.
 *
 * <p>This dataset implementation scans a directory for image files (supporting JPEG and PNG formats)
 * and loads them into {@link GradTensor} objects with the channel-height-width format {@code [C, H, W]}.
 * Images are loaded lazily on demand when {@link #get(int)} is called, enabling efficient memory usage
 * for large image collections.
 *
 * <p><b>Supported Formats:</b> JPG, JPEG, PNG (case-insensitive extensions)
 *
 * <p><b>Directory Structure:</b>
 * <pre>
 * image_directory/
 * ├── image1.jpg
 * ├── image2.png
 * ├── subdir/
 * │   ├── image3.jpeg
 * │   └── image4.jpg
 * └── ...
 * </pre>
 *
 * <p><b>Tensor Format:</b>
 * <ul>
 *   <li><b>C (Channels):</b> Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)</li>
 *   <li><b>H (Height):</b> Image height in pixels</li>
 *   <li><b>W (Width):</b> Image width in pixels</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>
 * Path imageDir = Paths.get("data/images");
 * ImageDataset dataset = new ImageDataset(imageDir);
 * 
 * // Get first image as tensor [3, 224, 224] for RGB images
 * GradTensor imageTensor = dataset.get(0);
 * 
 * // Get the original file path for metadata purposes
 * Path imagePath = dataset.getPath(0);
 * </pre>
 *
 * <p><b>Performance Characteristics:</b>
 * <ul>
 *   <li><b>Construction Time:</b> O(n) where n is total files in directory and subdirectories
 *                                  (performs single directory walk to collect image paths)</li>
 *   <li><b>Memory Usage:</b> O(n) for storing image file paths; actual image data loaded on demand</li>
 *   <li><b>Access Time:</b> O(1) lookup by index; O(k) image loading where k is image size</li>
 * </ul>
 *
 * <p><b>Error Handling:</b> If an image fails to load during {@link #get(int)}, a
 * {@link RuntimeException} is thrown containing the file path and original cause.
 *
 * @see GradTensor
 * @see Dataset
 */
public class ImageDataset implements Dataset<GradTensor> {

    private final List<Path> imagePaths;

    /**
     * Constructs an image dataset that scans the specified directory for image files.
     *
     * <p>This constructor recursively walks the directory tree, collecting all files with
     * supported image extensions (.jpg, .jpeg, .png). The collected paths are cached for
     * efficient subsequent access. Actual image data is not loaded until {@link #get(int)} is called.
     *
     * @param directory the path to the directory containing image files.
     *                  The directory is recursively scanned for images.
     *                  Must not be null and must be a valid directory.
     * @throws IOException if an I/O error occurs while reading the directory structure
     * @throws NullPointerException if {@code directory} is null
     */
    public ImageDataset(Path directory) throws IOException {
        try (var stream = Files.walk(directory)) {
            this.imagePaths = stream
                .filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.toString().toLowerCase();
                    return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
                })
                .collect(Collectors.toList());
        }
    }

    /**
     * Retrieves an image at the specified index as a gradient tensor.
     *
     * <p>The image is loaded from disk using {@link javax.imageio.ImageIO} and converted to a
     * {@link GradTensor} in channel-height-width format [C, H, W]. This method performs lazy
     * loading, meaning the image data is only read when explicitly requested.
     *
     * @param index the zero-based index of the image to retrieve
     * @return a {@link GradTensor} representing the image in [C, H, W] format
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     *         ({@code index < 0 || index >= size()})
     * @throws RuntimeException if the image file cannot be read or converted to tensor format,
     *         with the original {@link IOException} as cause
     */
    @Override
    public GradTensor get(int index) {
        Path path = imagePaths.get(index);
        try {
            // Converts [C, H, W]
            return GradTensor.fromImage(javax.imageio.ImageIO.read(path.toFile()));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load image: " + path, e);
        }
    }

    /**
     * Returns the total number of images in this dataset.
     *
     * @return the number of image files found in the directory and subdirectories
     */
    @Override
    public int size() {
        return imagePaths.size();
    }

    /**
     * Retrieves the file path of the image at the specified index.
     *
     * <p>This method is useful for obtaining metadata about the image source or for logging
     * and debugging purposes.
     *
     * @param index the zero-based index of the image
     * @return the {@link Path} to the image file
     * @throws IndexOutOfBoundsException if {@code index} is out of range
     *         ({@code index < 0 || index >= size()})
     */
    public Path getPath(int index) {
        return imagePaths.get(index);
    }
}
