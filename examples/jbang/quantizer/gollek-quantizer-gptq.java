///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek GPTQ Quantization Example
//
// Demonstrates GPTQ (GPT Quantization) for high-quality 4-bit model compression.
// GPTQ uses Hessian-based per-layer optimization for minimal accuracy loss.
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - GPTQ module built: cd gollek/core/quantizer/gollek-quantizer-gptq && mvn clean install
//
// Usage:
//   jbang gollek-quantizer-gptq.java --help
//   jbang gollek-quantizer-gptq.java --model /path/to/model.safetensors --output /path/to/quantized
//   jbang gollek-quantizer-gptq.java --demo  # Run with synthetic data
//
// DEPS tech.kayys.gollek:gollek-quantizer-gptq:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-safetensor-loader:0.1.0-SNAPSHOT
// DEPS info.picocli:picocli:4.7.5
// JAVA 25+

import tech.kayys.gollek.quantizer.gptq.GPTQQuantizerService;
import tech.kayys.gollek.quantizer.gptq.GPTQConfig;
import tech.kayys.gollek.quantizer.gptq.QuantizationResult;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.nio.file.Path;
import java.util.concurrent.Callable;

@Command(name = "gptq-quantizer", mixinStandardHelpOptions = true, version = "gptq-quantizer 0.1.0",
        description = "GPTQ 4-bit Quantization Example - Hessian-based per-layer optimization")
public class gollek_quantizer_gptq implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "Path to FP16/FP32 model directory")
    private Path modelPath;

    @Option(names = {"-o", "--output"}, description = "Output path for quantized model")
    private Path outputPath;

    @Option(names = {"--demo"}, description = "Run demo mode with synthetic data")
    private boolean demoMode;

    @Option(names = {"--bits"}, description = "Quantization bits (2, 3, 4, 8)", defaultValue = "4")
    private int bits;

    @Option(names = {"--group-size"}, description = "Group size for quantization", defaultValue = "128")
    private int groupSize;

    @Option(names = {"--damp"}, description = "Dampening percentage (0.0-1.0)", defaultValue = "0.01")
    private double dampPercent;

    @Option(names = {"--act-order"}, description = "Use activation ordering for better quality")
    private boolean actOrder;

    @Option(names = {"--symmetric"}, description = "Use symmetric quantization")
    private boolean symmetric;

    @Option(names = {"--num-samples"}, description = "Number of calibration samples", defaultValue = "128")
    private int numSamples;

    @Option(names = {"--seq-len"}, description = "Sequence length for calibration", defaultValue = "2048")
    private int seqLen;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new gollek_quantizer_gptq()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek GPTQ Quantization Example                ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        if (demoMode || modelPath == null) {
            runDemoMode();
            return 0;
        }

        if (outputPath == null) {
            outputPath = modelPath.resolveSibling(modelPath.getFileName() + "-gptq-" + bits + "bit");
        }

        // Build GPTQ configuration
        GPTQConfig config = GPTQConfig.builder()
                .bits(bits)
                .groupSize(groupSize)
                .dampPercent(dampPercent)
                .actOrder(actOrder)
                .symmetric(symmetric)
                .numSamples(numSamples)
                .seqLen(seqLen)
                .build();

        System.out.println("Configuration:");
        System.out.println("  Model:         " + modelPath);
        System.out.println("  Output:        " + outputPath);
        System.out.println("  Bits:          " + bits);
        System.out.println("  Group Size:    " + groupSize);
        System.out.println("  Dampening:     " + dampPercent);
        System.out.println("  ActOrder:      " + actOrder);
        System.out.println("  Symmetric:     " + symmetric);
        System.out.println("  Calib Samples: " + numSamples);
        System.out.println("  Seq Length:    " + seqLen);
        System.out.println();

        // Run quantization
        try (GPTQQuantizerService service = new GPTQQuantizerService()) {
            System.out.println("Starting GPTQ quantization...");
            long startTime = System.currentTimeMillis();

            try {
                QuantizationResult result = service.quantize(modelPath, outputPath, config);

                long elapsed = System.currentTimeMillis() - startTime;

                System.out.println();
                System.out.println("✓ Quantization completed successfully!");
                System.out.println();
                System.out.println("Results:");
                System.out.println("  Input Tensors:     " + result.inputTensors());
                System.out.println("  Output Tensors:    " + result.outputTensors());
                System.out.println("  Input Size:        " + formatBytes(result.inputSizeBytes()));
                System.out.println("  Output Size:       " + formatBytes(result.outputSizeBytes()));
                System.out.println("  Compression Ratio: " + String.format("%.2f", result.compressionRatio()) + "x");
                System.out.println("  Throughput:        " + String.format("%.1f", result.throughputMBps()) + " MB/s");
                System.out.println("  Elapsed Time:      " + elapsed + "ms");
                System.out.println();
                System.out.println("Quantized model saved to: " + outputPath);

            } catch (Exception e) {
                System.err.println("❌ Quantization failed: " + e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }

        return 0;
    }

    /**
     * Run demo mode with synthetic data to show GPTQ capabilities.
     */
    private void runDemoMode() {
        System.out.println("Running in demo mode (no real model required)");
        System.out.println();

        // Show configuration examples
        System.out.println("GPTQ Configuration Examples:");
        System.out.println();

        // Example 1: Standard 4-bit
        GPTQConfig config4bit = GPTQConfig.gptq4bit();
        System.out.println("1. Standard 4-bit GPTQ:");
        System.out.println("   Bits:       " + config4bit.bits());
        System.out.println("   Group Size: " + config4bit.groupSize());
        System.out.println("   Dampening:  " + config4bit.dampPercent());
        System.out.println("   ActOrder:   " + config4bit.actOrder());
        System.out.println();

        // Example 2: High quality (group size 64)
        GPTQConfig configHQ = GPTQConfig.builder().bits(4).groupSize(64).build();
        System.out.println("2. High Quality 4-bit (group size 64):");
        System.out.println("   Bits:       " + configHQ.bits());
        System.out.println("   Group Size: " + configHQ.groupSize());
        System.out.println("   Expected quality improvement: ~10-15%");
        System.out.println();

        // Example 3: 8-bit for minimal quality loss
        GPTQConfig config8bit = GPTQConfig.gptq8bit();
        System.out.println("3. 8-bit GPTQ (minimal quality loss):");
        System.out.println("   Bits:       " + config8bit.bits());
        System.out.println("   Group Size: " + config8bit.groupSize());
        System.out.println("   Expected compression: 4x");
        System.out.println();

        // Example 4: 3-bit for maximum compression
        GPTQConfig config3bit = GPTQConfig.gptq3bit();
        System.out.println("4. 3-bit GPTQ (maximum compression):");
        System.out.println("   Bits:       " + config3bit.bits());
        System.out.println("   Group Size: " + config3bit.groupSize());
        System.out.println("   Expected compression: 10x");
        System.out.println("   Note: Higher quality loss than 4-bit");
        System.out.println();

        System.out.println("To quantize a real model:");
        System.out.println("  jbang gollek-quantizer-gptq.java \\");
        System.out.println("    --model /path/to/model.safetensors \\");
        System.out.println("    --output /path/to/quantized \\");
        System.out.println("    --bits 4 \\");
        System.out.println("    --group-size 128");
        System.out.println();

        // Simulate quantization progress
        System.out.println("Simulating quantization progress...");
        String[] phases = {"Loading model", "Collecting activations", "Quantizing layer 1/32",
                "Quantizing layer 16/32", "Quantizing layer 32/32", "Saving quantized model"};

        for (int i = 0; i < phases.length; i++) {
            System.out.printf("  [%3d%%] %s%n", (i + 1) * 100 / phases.length, phases[i]);
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        System.out.println();
        System.out.println("✓ Demo complete!");
        System.out.println();
        System.out.println("Expected Results for LLaMA-7B:");
        System.out.println("  Original Size:     13.5 GB (FP16)");
        System.out.println("  Quantized Size:    3.8 GB (4-bit)");
        System.out.println("  Compression:       3.5x");
        System.out.println("  Quality Loss:      <1% perplexity increase");
        System.out.println("  Quantization Time: ~25 minutes");
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
