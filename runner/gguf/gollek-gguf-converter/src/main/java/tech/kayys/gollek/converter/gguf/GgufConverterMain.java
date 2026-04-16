package tech.kayys.gollek.converter.gguf;

import java.io.IOException;
import java.nio.file.*;
import java.util.Arrays;

/**
 * CLI entry point for the GGUF converter.
 *
 * <pre>
 * Usage:
 *   java -jar gguf-converter.jar convert  &lt;hf-dir&gt; &lt;output.gguf&gt; [OPTIONS]
 *   java -jar gguf-converter.jar inspect  &lt;file.gguf&gt;
 *
 * Options for convert:
 *   --type  &lt;F16|F32|Q8_0|Q4_0&gt;  Quantization target (default: F16)
 *   --version &lt;str&gt;              Model version string (default: 1.0)
 *   --verbose                    Enable verbose logging
 * </pre>
 */
public final class GgufConverterMain {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            printHelp();
            return;
        }

        switch (args[0]) {
            case "convert" -> runConvert(Arrays.copyOfRange(args, 1, args.length));
            case "inspect" -> runInspect(Arrays.copyOfRange(args, 1, args.length));
            default -> {
                System.err.println("Unknown command: " + args[0]);
                printHelp();
            }
        }
    }

    // ── convert ───────────────────────────────────────────────────────────

    private static void runConvert(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: convert <hf-dir> <output.gguf> [--type F16] [--version 1.0] [--verbose]");
            System.exit(1);
        }

        Path inputDir = Path.of(args[0]);
        Path outputFile = Path.of(args[1]);
        GgmlType quantType = GgmlType.F16;
        String version = "1.0";
        boolean verbose = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--type" -> {
                    quantType = GgmlType.fromLabel(args[++i]);
                }
                case "--version" -> {
                    version = args[++i];
                }
                case "--verbose" -> {
                    verbose = true;
                }
                default -> {
                    System.err.println("Unknown option: " + args[i]);
                    System.exit(1);
                }
            }
        }

        if (!Files.isDirectory(inputDir)) {
            System.err.println("Input directory does not exist: " + inputDir);
            System.exit(1);
        }

        var opts = new HfToGgufConverter.ConvertOptions(
                inputDir, outputFile, quantType, version, verbose);
        HfToGgufConverter.convert(opts);
    }

    // ── inspect ───────────────────────────────────────────────────────────

    private static void runInspect(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: inspect <file.gguf>");
            System.exit(1);
        }
        Path p = Path.of(args[0]);
        if (!Files.exists(p)) {
            System.err.println("File not found: " + p);
            System.exit(1);
        }
        try (GgufReader reader = new GgufReader(p)) {
            GgufModel model = reader.read();

            System.out.println("══════════════════════════════════════════════");
            System.out.println("GGUF File: " + p.getFileName());
            System.out.println("Architecture : " + model.architecture());
            System.out.println("Model name   : " + model.modelName());
            System.out.println("Alignment    : " + model.alignment());
            System.out.println("Tensors      : " + model.tensors().size());
            System.out.println("Metadata KVs : " + model.metadata().size());
            System.out.println("══════════════════════════════════════════════");

            System.out.println("\n── Metadata ──");
            model.metadata().forEach((k, v) -> {
                String display = switch (v) {
                    case GgufMetaValue.ArrayVal av ->
                        "[" + av.elementType() + " × " + av.elements().size() + "]";
                    case GgufMetaValue.StringVal sv ->
                        '"' + truncate(sv.value(), 80) + '"';
                    default -> v.toString();
                };
                System.out.printf("  %-50s = %s%n", k, display);
            });

            System.out.println("\n── Tensors ──");
            long totalBytes = 0;
            for (TensorInfo t : model.tensors()) {
                long bytes = t.dataSize();
                totalBytes += bytes;
                System.out.printf("  %-50s  %-8s  %-20s  %,d bytes%n",
                        t.name(), t.type().label,
                        java.util.Arrays.toString(t.ne()), bytes);
            }
            System.out.printf("%nTotal tensor data: %,.1f MB%n", totalBytes / 1e6);
        }
    }

    // ── Help ──────────────────────────────────────────────────────────────

    private static void printHelp() {
        System.out.println("""
                GGUF Converter — pure Java, JDK 25 FFM API
                ───────────────────────────────────────────
                Commands:
                  convert <hf-dir> <output.gguf> [OPTIONS]
                      --type    F32|F16|Q8_0|Q4_0   (default: F16)
                      --version <str>               (default: 1.0)
                      --verbose

                  inspect <file.gguf>
                """);
    }

    private static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max - 3) + "...";
    }
}
