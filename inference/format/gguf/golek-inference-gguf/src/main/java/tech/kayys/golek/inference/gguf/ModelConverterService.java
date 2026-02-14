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

    private Path scriptPath;

    @PostConstruct
    void init() {
        resolveScriptPath();
    }

    public boolean isAvailable() {
        return scriptPath != null && Files.exists(scriptPath);
    }

    public void convert(Path inputDir, Path outputFile) throws IOException, InterruptedException {
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

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Process process = pb.start();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                LOG.debugf("[convert] %s", line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
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
}
