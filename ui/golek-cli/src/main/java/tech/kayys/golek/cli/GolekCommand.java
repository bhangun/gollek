package tech.kayys.golek.cli;

import io.quarkus.picocli.runtime.annotations.TopCommand;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import picocli.CommandLine.Command;
import tech.kayys.golek.cli.commands.ChatCommand;
import tech.kayys.golek.cli.commands.DeleteCommand;
import tech.kayys.golek.cli.commands.ExtensionsCommand;
import tech.kayys.golek.cli.commands.InfoCommand;
import tech.kayys.golek.cli.commands.ListCommand;
import tech.kayys.golek.cli.commands.ProvidersCommand;
import tech.kayys.golek.cli.commands.PullCommand;
import tech.kayys.golek.cli.commands.RunCommand;
import tech.kayys.golek.cli.commands.SafetensorsCommand;
import tech.kayys.golek.cli.commands.ShowCommand;

import picocli.CommandLine;
import picocli.CommandLine.Option;

@TopCommand
@Command(name = "golek", mixinStandardHelpOptions = true, version = "1.0.0", description = "Golek Inference CLI - Run local and cloud AI models", subcommands = {
        RunCommand.class,
        ChatCommand.class,
        PullCommand.class,
        ListCommand.class,
        ShowCommand.class,
        DeleteCommand.class,
        ProvidersCommand.class,
        ExtensionsCommand.class,
        InfoCommand.class,
        SafetensorsCommand.class
})
public class GolekCommand implements Runnable {
    private static final String HF_TOKEN_PROPERTY = "wayang.inference.repository.huggingface.token";
    private static final String GGUF_LIB_DIR_PROPERTY = "gguf.provider.native.library-dir";
    private static final List<String> HF_TOKEN_KEYS = List.of(
            "WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            HF_TOKEN_PROPERTY);

    @Option(names = { "--log" }, description = "Enable verbose logging", scope = CommandLine.ScopeType.INHERIT)
    boolean verbose;

    public GolekCommand() {
        configureHuggingFaceTokenFromDotEnv();
        configureGgufNativeLibraryDir();
    }

    @Override
    public void run() {
        if (verbose) {
            System.setProperty("quarkus.log.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.golek\".level", "DEBUG");
            System.setProperty("gguf.provider.verbose-logging", "true");
            // Workaround for programmatic change during runtime if possible,
            // but Picocli runs before Quarkus finishes all init in some modes.
            // For now, these system properties might help, or we check this flag in
            // commands.
        }
    }

    private void configureHuggingFaceTokenFromDotEnv() {
        if (hasText(System.getProperty(HF_TOKEN_PROPERTY))) {
            return;
        }

        // First map process environment variables to the Quarkus property key.
        for (String key : HF_TOKEN_KEYS) {
            String envValue = System.getenv(key);
            if (hasText(envValue)) {
                System.setProperty(HF_TOKEN_PROPERTY, envValue.trim());
                return;
            }
        }

        // Fallback to .env in current working directory.
        Path dotEnvPath = Path.of(".env");
        if (!Files.isRegularFile(dotEnvPath)) {
            return;
        }

        try {
            Map<String, String> dotEnv = parseDotEnv(dotEnvPath);
            for (String key : HF_TOKEN_KEYS) {
                String value = dotEnv.get(key);
                if (hasText(value)) {
                    System.setProperty(HF_TOKEN_PROPERTY, value.trim());
                    return;
                }
            }
        } catch (IOException ignored) {
            // Ignore .env parse failures and continue normal startup.
        }
    }

    private static Map<String, String> parseDotEnv(Path path) throws IOException {
        Map<String, String> values = new HashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (String rawLine : lines) {
            String line = rawLine == null ? "" : rawLine.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            if (line.startsWith("export ")) {
                line = line.substring("export ".length()).trim();
            }
            int separator = line.indexOf('=');
            if (separator <= 0) {
                continue;
            }
            String key = line.substring(0, separator).trim();
            String value = line.substring(separator + 1).trim();
            if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
                value = value.substring(1, value.length() - 1);
            }
            values.put(key, value);
        }
        return values;
    }

    private static boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }

    private void configureGgufNativeLibraryDir() {
        if (hasText(System.getProperty(GGUF_LIB_DIR_PROPERTY)) || hasText(System.getenv("GOLEK_LLAMA_LIB_DIR"))) {
            return;
        }

        Path cwd = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        String[] candidates = {
                "extension/format/gguf/golek-ext-format-gguf/target/llama-cpp/lib",
                "inference/format/gguf/source/llama-cpp/llama.cpp/build/bin"
        };

        Path current = cwd;
        for (int i = 0; i < 8 && current != null; i++) {
            for (String candidate : candidates) {
                Path dir = current.resolve(candidate);
                if (Files.isDirectory(dir)) {
                    System.setProperty(GGUF_LIB_DIR_PROPERTY, dir.toString());
                    return;
                }
            }
            current = current.getParent();
        }
    }
}
