package tech.kayys.golek.inference.gguf;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Service to convert HuggingFace models to GGUF using llama.cpp scripts.
 */
@ApplicationScoped
public class ModelConverterService {

    private static final Logger LOG = Logger.getLogger(ModelConverterService.class);
    private static final String CONVERT_SCRIPT = "convert_hf_to_gguf.py";

    @ConfigProperty(name = "golek.gguf.converter.script")
    Optional<String> configuredScriptPath;

    @ConfigProperty(name = "golek.gguf.python.command", defaultValue = "python3")
    String pythonCommand;

    @ConfigProperty(name = "golek.gguf.converter.outtype")
    Optional<String> configuredOutType;

    private Path scriptPath;

    @PostConstruct
    void init() {
        resolveScriptPath();
    }

    public boolean isAvailable() {
        return scriptPath != null && Files.exists(scriptPath);
    }

    public void convert(Path inputDir, Path outputFile) throws IOException, InterruptedException {
        convert(inputDir, outputFile, configuredOutType.orElse(null));
    }

    public void convert(Path inputDir, Path outputFile, String outType) throws IOException, InterruptedException {
        if (!isAvailable()) {
            throw new IllegalStateException("Conversion script not found");
        }

        LOG.infof("Converting model from %s to %s", inputDir, outputFile);

        List<String> command = new ArrayList<>();
        command.add(pythonCommand);
        command.add(scriptPath.toAbsolutePath().toString());
        command.add(inputDir.toAbsolutePath().toString());
        command.add("--outfile");
        command.add(outputFile.toAbsolutePath().toString());
        String normalizedOutType = normalizeOutType(outType);
        if (normalizedOutType != null) {
            command.add("--outtype");
            command.add(normalizedOutType);
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Process process = pb.start();
        StringBuilder recentOutput = new StringBuilder();
        int linesCaptured = 0;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                LOG.debugf("[convert] %s", line);
                if (linesCaptured < 120) {
                    if (recentOutput.length() > 0) {
                        recentOutput.append('\n');
                    }
                    recentOutput.append(line);
                    linesCaptured++;
                } else if (linesCaptured == 120) {
                    recentOutput.append("\n...");
                    linesCaptured++;
                }
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            String output = recentOutput.toString().trim();
            if (output.contains("ModuleNotFoundError: No module named 'transformers'")) {
                throw new IOException("Python dependency missing: transformers. Install converter dependencies with: "
                        + pythonCommand + " -m pip install --upgrade transformers sentencepiece safetensors");
            }
            if (!output.isEmpty()) {
                throw new IOException("Conversion process failed with exit code " + exitCode + ": " + output);
            }
            throw new IOException("Conversion process failed with exit code " + exitCode);
        }

        LOG.infof("Conversion completed successfully: %s", outputFile);
    }

    private void resolveScriptPath() {
        // 1. Check configuration
        if (configuredScriptPath.isPresent()) {
            Path path = Paths.get(configuredScriptPath.get());
            if (Files.exists(path)) {
                scriptPath = path;
                LOG.infof("Using configured conversion script: %s", scriptPath);
                return;
            }
            LOG.warnf("Configured conversion script not found: %s", path);
        }

        String envScript = System.getenv("GOLEK_GGUF_CONVERTER_SCRIPT");
        if (envScript != null && !envScript.isBlank()) {
            Path path = Paths.get(envScript.trim());
            if (Files.exists(path)) {
                scriptPath = path;
                LOG.infof("Using converter script from GOLEK_GGUF_CONVERTER_SCRIPT: %s", scriptPath);
                return;
            }
        }

        // 2. Check common relative paths from current working directory and parents.
        Path cwd = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        String[] relativeCandidates = {
                "extension/format/gguf/vendor/llama-cpp/llama.cpp/convert_hf_to_gguf.py",
                "inference/format/gguf/source/llama-cpp/llama.cpp/convert_hf_to_gguf.py"
        };
        Path current = cwd;
        for (int i = 0; i < 8 && current != null; i++) {
            for (String relative : relativeCandidates) {
                Path candidate = current.resolve(relative);
                if (Files.exists(candidate)) {
                    scriptPath = candidate;
                    LOG.infof("Found conversion script at: %s", scriptPath);
                    return;
                }
            }
            current = current.getParent();
        }

        // 2. Search in common locations
        // This is a heuristic for development environments
        String projectRoot = System.getProperty("user.dir");
        try (Stream<Path> paths = Files.walk(Paths.get(projectRoot))) {
            Optional<Path> found = paths
                    .filter(p -> p.getFileName().toString().equals(CONVERT_SCRIPT))
                    .filter(p -> p.toString().contains("llama.cpp"))
                    .findFirst();

            if (found.isPresent()) {
                scriptPath = found.get();
                LOG.infof("Found conversion script: %s", scriptPath);
                return;
            }
        } catch (IOException e) {
            LOG.debug("Error searching for conversion script", e);
        }

        LOG.warn("Could not find convert_hf_to_gguf.py script. Model conversion will not be available.");
    }

    private String normalizeOutType(String outType) {
        if (outType == null) {
            return null;
        }
        String normalized = outType.trim().toLowerCase();
        if (normalized.isEmpty()) {
            return null;
        }
        return normalized;
    }
}
