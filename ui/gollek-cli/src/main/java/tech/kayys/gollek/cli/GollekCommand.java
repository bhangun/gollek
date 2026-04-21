package tech.kayys.gollek.cli;

import io.quarkus.picocli.runtime.annotations.TopCommand;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import picocli.CommandLine.Command;
import tech.kayys.gollek.cli.commands.ChatCommand;
import tech.kayys.gollek.cli.commands.ConvertCommand;
import tech.kayys.gollek.cli.commands.DeleteCommand;
import tech.kayys.gollek.cli.commands.ExtensionsCommand;
import tech.kayys.gollek.cli.commands.InfoCommand;
import tech.kayys.gollek.cli.commands.ListCommand;
import tech.kayys.gollek.cli.commands.McpCommand;
import tech.kayys.gollek.cli.commands.ProvidersCommand;
import tech.kayys.gollek.cli.commands.PullCommand;
import tech.kayys.gollek.cli.commands.RunCommand;
import tech.kayys.gollek.cli.commands.SafetensorsCommand;
import tech.kayys.gollek.cli.commands.ShowCommand;
import tech.kayys.gollek.cli.commands.EmbedCommand;
import tech.kayys.gollek.cli.commands.BatchCommand;
import tech.kayys.gollek.cli.commands.JobsCommand;
import tech.kayys.gollek.cli.commands.StatsCommand;
import tech.kayys.gollek.cli.commands.LogsCommand;
import tech.kayys.gollek.cli.commands.RegisterCommand;
import tech.kayys.gollek.cli.commands.MultimodalCommand;
import tech.kayys.gollek.cli.commands.LiteRTCommand;
import tech.kayys.gollek.cli.commands.OnnxCommand;
import tech.kayys.gollek.cli.commands.QuantizeCommand;

import picocli.CommandLine;
import picocli.CommandLine.Option;

@TopCommand
@Command(name = "gollek", mixinStandardHelpOptions = true, version = "1.0.0", description = "Gollek Inference CLI - Run local and cloud AI models", subcommands = {
        RunCommand.class,
        ChatCommand.class,
        PullCommand.class,
        ConvertCommand.class,
        ListCommand.class,
        McpCommand.class,
        ShowCommand.class,
        DeleteCommand.class,
        ProvidersCommand.class,
        ExtensionsCommand.class,
        InfoCommand.class,
        SafetensorsCommand.class,
        EmbedCommand.class,
        BatchCommand.class,
        JobsCommand.class,
        StatsCommand.class,
        LogsCommand.class,
        RegisterCommand.class,
        MultimodalCommand.class,
        LiteRTCommand.class,
        OnnxCommand.class,
        QuantizeCommand.class
})

public class GollekCommand implements Runnable {
    private static final org.jboss.logging.Logger LOG = org.jboss.logging.Logger.getLogger(GollekCommand.class);
    private static final String HF_TOKEN_PROPERTY = "wayang.inference.repository.huggingface.token";
    private static final String GGUF_LIB_DIR_PROPERTY = "gguf.provider.native.library-dir";
    private static final String MCP_SERVERS_JSON_PROPERTY = "wayang.inference.mcp.mcp-servers-json";
    private static final String MCP_SERVERS_JSON_FILE_PROPERTY = "wayang.inference.mcp.mcp-servers-json-file";
    private static final String GOLLEK_MCP_SERVERS_JSON = "GOLLEK_MCP_SERVERS_JSON";
    private static final String GOLLEK_MCP_SERVERS_FILE = "GOLLEK_MCP_SERVERS_FILE";
    private static final Path DEFAULT_MCP_REGISTRY_FILE = GollekHome.path("mcp", "servers.json");
    private static final List<String> HF_TOKEN_KEYS = List.of(
            "WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            HF_TOKEN_PROPERTY);

    @Option(names = { "--log" }, description = "Enable verbose logging", scope = CommandLine.ScopeType.INHERIT)
    public boolean verbose;

    @Option(names = {
            "--mcp-servers-json" }, description = "Inline MCP config JSON with `mcpServers` object", scope = CommandLine.ScopeType.INHERIT)
    String mcpServersJson;

    @Option(names = {
            "--mcp-servers-file" }, description = "Path to MCP config JSON file containing `mcpServers`", scope = CommandLine.ScopeType.INHERIT)
    String mcpServersFile;

    @Option(names = {
            "--use-cpu" }, description = "Use CPU instead of GPU (disable GPU acceleration)", scope = CommandLine.ScopeType.INHERIT)
    boolean useCpu;

    @Option(names = {
            "--enable-cpu" }, description = "Enable CPU fallback (use CPU if GPU not available)", scope = CommandLine.ScopeType.INHERIT)
    boolean enableCpu;

    @Option(names = {
            "--platform" }, description = "Force specific kernel platform (metal, cuda, rocm, directml, cpu)", scope = CommandLine.ScopeType.INHERIT)
    String platform;

    public GollekCommand() {
        GollekHome.applySystemProperties();
        // Enforce silent logging by default as early as possible
        System.setProperty("quarkus.log.level", "WARN");
        System.setProperty("quarkus.log.console.level", "WARN");
        System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "WARN");
        System.setProperty("quarkus.log.category.\"tech.kayys.gollek.sdk\".level", "WARN");
        System.setProperty("quarkus.log.category.\"io.quarkus\".level", "WARN");

        configureFileLoggingFromEnvironment();
        configureHuggingFaceTokenFromDotEnv();
        configureGgufNativeLibraryDir();
        configureMcpServersFromEnvironmentAndArgs();

        // Configure kernel platform
        configureKernelPlatform();
    }

    /**
     * Configure kernel platform based on CLI options.
     */
    private void configureKernelPlatform() {
        if (useCpu) {
            System.setProperty("gollek.kernel.force.cpu", "true");
            System.out.println("⚠️  CPU usage enabled (GPU acceleration disabled)");
        }

        if (enableCpu) {
            System.setProperty("gollek.kernel.cpu.fallback", "true");
            LOG.info("CPU fallback enabled (will use CPU if GPU not available)");
        }

        if (platform != null && !platform.trim().isEmpty()) {
            System.setProperty("gollek.kernel.platform", platform.trim().toLowerCase());
            System.out.printf("⚠️  Kernel platform forced to: %s%n", platform.trim().toLowerCase());
        }
    }

    @Override
    public void run() {
        applyRuntimeOverrides();
        if (verbose) {
            System.setProperty("quarkus.log.level", "DEBUG");
            System.setProperty("quarkus.log.console.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.gollek.inference.libtorch\".level", "DEBUG");
            System.setProperty("gguf.provider.verbose-logging", "true");
            // Workaround for programmatic change during runtime if possible,
            // but Picocli runs before Quarkus finishes all init in some modes.
            // For now, these system properties might help, or we check this flag in
            // commands.
        }
    }

    public void applyRuntimeOverrides() {
        if (hasText(mcpServersJson)) {
            System.setProperty(MCP_SERVERS_JSON_PROPERTY, mcpServersJson.trim());
        }
        if (hasText(mcpServersFile)) {
            System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, mcpServersFile.trim());
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

    private void configureFileLoggingFromEnvironment() {
        String fileLogging = System.getenv("GOLLEK_CLI_FILE_LOG_ENABLED");
        if (hasText(fileLogging)) {
            System.setProperty("quarkus.log.file.enabled", fileLogging.trim());
        }
        String logFilePath = System.getenv("GOLLEK_CLI_LOG_FILE");
        if (hasText(logFilePath)) {
            System.setProperty("quarkus.log.file.path", logFilePath.trim());
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
        if (hasText(System.getProperty(GGUF_LIB_DIR_PROPERTY)) || hasText(System.getenv("GOLLEK_LLAMA_LIB_DIR"))) {
            return;
        }

        Path cwd = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        String[] candidates = {
                GollekHome.path("libs", "llama").toString(),
                GollekHome.path("source", "vendor", "llama.cpp", "build", "bin").toString()
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

    private void configureMcpServersFromEnvironmentAndArgs() {
        if (!hasText(System.getProperty(MCP_SERVERS_JSON_PROPERTY))) {
            String envJson = System.getenv(GOLLEK_MCP_SERVERS_JSON);
            if (hasText(envJson)) {
                System.setProperty(MCP_SERVERS_JSON_PROPERTY, envJson.trim());
            }
        }

        if (!hasText(System.getProperty(MCP_SERVERS_JSON_FILE_PROPERTY))) {
            String envFile = System.getenv(GOLLEK_MCP_SERVERS_FILE);
            if (hasText(envFile)) {
                System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, envFile.trim());
            }
        }

        Path dotEnvPath = Path.of(".env");
        if (Files.isRegularFile(dotEnvPath)) {
            try {
                Map<String, String> dotEnv = parseDotEnv(dotEnvPath);
                if (!hasText(System.getProperty(MCP_SERVERS_JSON_PROPERTY))) {
                    String dotEnvJson = dotEnv.get(GOLLEK_MCP_SERVERS_JSON);
                    if (hasText(dotEnvJson)) {
                        System.setProperty(MCP_SERVERS_JSON_PROPERTY, dotEnvJson.trim());
                    }
                }
                if (!hasText(System.getProperty(MCP_SERVERS_JSON_FILE_PROPERTY))) {
                    String dotEnvFile = dotEnv.get(GOLLEK_MCP_SERVERS_FILE);
                    if (hasText(dotEnvFile)) {
                        System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, dotEnvFile.trim());
                    }
                }
            } catch (IOException ignored) {
                // Ignore .env parse failures and continue normal startup.
            }
        }

        List<String> args = parseCommandLineArgs(System.getProperty("sun.java.command", ""));
        for (int i = 0; i < args.size(); i++) {
            String arg = args.get(i);
            if (arg.startsWith("--mcp-servers-json=")) {
                String value = arg.substring("--mcp-servers-json=".length()).trim();
                if (hasText(value)) {
                    System.setProperty(MCP_SERVERS_JSON_PROPERTY, value);
                }
                continue;
            }
            if (arg.equals("--mcp-servers-json") && i + 1 < args.size()) {
                String value = args.get(++i);
                if (hasText(value)) {
                    System.setProperty(MCP_SERVERS_JSON_PROPERTY, value.trim());
                }
                continue;
            }
            if (arg.startsWith("--mcp-servers-file=")) {
                String value = arg.substring("--mcp-servers-file=".length()).trim();
                if (hasText(value)) {
                    System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, value);
                }
                continue;
            }
            if (arg.equals("--mcp-servers-file") && i + 1 < args.size()) {
                String value = args.get(++i);
                if (hasText(value)) {
                    System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, value.trim());
                }
            }
        }

        if (!hasText(System.getProperty(MCP_SERVERS_JSON_PROPERTY))
                && !hasText(System.getProperty(MCP_SERVERS_JSON_FILE_PROPERTY))
                && Files.isRegularFile(DEFAULT_MCP_REGISTRY_FILE)) {
            System.setProperty(MCP_SERVERS_JSON_FILE_PROPERTY, DEFAULT_MCP_REGISTRY_FILE.toString());
        }
    }

    private static List<String> parseCommandLineArgs(String raw) {
        List<String> tokens = new java.util.ArrayList<>();
        if (!hasText(raw)) {
            return tokens;
        }

        StringBuilder current = new StringBuilder();
        boolean inSingle = false;
        boolean inDouble = false;
        boolean escaped = false;
        for (int i = 0; i < raw.length(); i++) {
            char ch = raw.charAt(i);
            if (escaped) {
                current.append(ch);
                escaped = false;
                continue;
            }
            if (ch == '\\') {
                escaped = true;
                continue;
            }
            if (ch == '\'' && !inDouble) {
                inSingle = !inSingle;
                continue;
            }
            if (ch == '"' && !inSingle) {
                inDouble = !inDouble;
                continue;
            }
            if (Character.isWhitespace(ch) && !inSingle && !inDouble) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                continue;
            }
            current.append(ch);
        }
        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        return tokens;
    }
}
