package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.function.UnaryOperator;
import javax.imageio.ImageIO;

/**
 * Image classification dataset — loads images from a directory structure.
 *
 * <p>Expected layout:
 * <pre>
 *   root/
 *     class_a/  img1.jpg  img2.png  ...
 *     class_b/  img3.jpg  ...
 * </pre>
 *
 * <p>Images are loaded as {@code [C, H, W]} float tensors in [0, 1].
 * Supports JPEG, PNG, BMP, GIF via {@link ImageIO}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var transform = Transforms.compose(
 *     Transforms.randomHorizontalFlip(0.5f),
 *     Transforms.normalize(new float[]{0.485f,0.456f,0.406f},
 *                          new float[]{0.229f,0.224f,0.225f})
 * );
 * var dataset = new ImageDataset(Path.of("imagenet/train"), transform);
 * var loader  = DataLoader.builder(dataset).batchSize(32).shuffle(true).build();
 * }</pre>
 */
public final class ImageDataset implements Dataset {

    private final List<Path>   imagePaths;
    private final List<Integer> labels;
    private final List<String>  classNames;
    private final UnaryOperator<GradTensor> transform;

    /**
     * Creates an image dataset from a root directory.
     *
     * @param root      root directory with one subdirectory per class
     * @param transform optional transform applied to each image (can be null)
     * @throws IOException if the directory cannot be read
     */
    public ImageDataset(Path root, UnaryOperator<GradTensor> transform) throws IOException {
        this.transform  = transform;
        this.imagePaths = new ArrayList<>();
        this.labels     = new ArrayList<>();
        this.classNames = new ArrayList<>();

        List<Path> classDirs = new ArrayList<>();
        try (var stream = Files.list(root)) {
            stream.filter(Files::isDirectory).sorted().forEach(classDirs::add);
        }

        for (int c = 0; c < classDirs.size(); c++) {
            classNames.add(classDirs.get(c).getFileName().toString());
            final int label = c;
            try (var files = Files.walk(classDirs.get(c), 1)) {
                files.filter(p -> isImage(p.toString())).forEach(p -> {
                    imagePaths.add(p);
                    labels.add(label);
                });
            }
        }
    }

    /** Creates an image dataset without transforms. */
    public ImageDataset(Path root) throws IOException { this(root, null); }

    @Override public int size() { return imagePaths.size(); }

    /**
     * Loads and returns the image at the given index.
     *
     * <p>Images are loaded lazily (not cached) to support large datasets.
     *
     * @param index sample index
     * @return {@link Dataset.Sample} with image tensor {@code [3, H, W]} and label scalar
     */
    @Override
    public Sample get(int index) {
        try {
            BufferedImage img = ImageIO.read(imagePaths.get(index).toFile());
            if (img == null) return emptyRgb();
            GradTensor tensor = GradTensor.fromImage(img);
            if (transform != null) tensor = transform.apply(tensor);
            return new Sample(tensor, GradTensor.scalar(labels.get(index)));
        } catch (IOException e) {
            return emptyRgb();
        }
    }

    /** @return list of class names in alphabetical order */
    public List<String> classNames() { return Collections.unmodifiableList(classNames); }

    /** @return number of classes */
    public int numClasses() { return classNames.size(); }

    private static boolean isImage(String path) {
        String p = path.toLowerCase();
        return p.endsWith(".jpg") || p.endsWith(".jpeg")
            || p.endsWith(".png") || p.endsWith(".bmp") || p.endsWith(".gif");
    }

    private static Sample emptyRgb() {
        return new Sample(GradTensor.zeros(3, 1, 1), GradTensor.scalar(0));
    }
}
