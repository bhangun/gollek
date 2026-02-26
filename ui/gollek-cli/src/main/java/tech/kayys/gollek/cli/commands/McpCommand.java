package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import tech.kayys.gollek.sdk.local.GollekLocalClient;
import tech.kayys.gollek.sdk.mcp.McpAddRequest;
import tech.kayys.gollek.sdk.mcp.McpEditRequest;
import tech.kayys.gollek.sdk.mcp.McpDoctorReport;
import tech.kayys.gollek.sdk.mcp.McpTestEntry;
import tech.kayys.gollek.sdk.mcp.McpTestReport;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.mcp.registry.McpRegistryEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.BatchInferenceRequest;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import io.smallrye.mutiny.Multi;
import java.util.concurrent.CompletableFuture;
import jakarta.json.Json;
import jakarta.json.JsonObject;
import jakarta.json.JsonReader;
import jakarta.json.JsonValue;

import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Map;

import java.util.List;

@Dependent
@Unremovable
@Command(name = "mcp", description = "Manage MCP server registry", subcommands = {
        McpCommand.Add.class,
        McpCommand.Remove.class,
        McpCommand.RenameServer.class,
        McpCommand.EditServer.class,
        McpCommand.EnableServer.class,
        McpCommand.DisableServer.class,
        McpCommand.ListServers.class,
        McpCommand.ShowServer.class,
        McpCommand.DoctorServers.class,
        McpCommand.ImportServers.class,
        McpCommand.ExportServers.class,
        McpCommand.TestServer.class
})
public class McpCommand implements Runnable {
    private static final String LIST_ONLY_MARKER = "__MCP_LIST_ONLY__";

    @Override
    public void run() {
        System.out.println(
                "Use one of: gollek mcp add | remove | rename | edit | enable | disable | list | show | doctor | import | export | test");
    }

    static abstract class McpSubcommand {
        @Inject
        GollekLocalClient localClient;

        GollekLocalClient client() {
            if (localClient == null) {
                // Fallback for non-CDI environments (e.g. McpCommandTest)
                return new GollekLocalClient() {
                    private final McpRegistryManager registry = new McpRegistryEngine();

                    @Override
                    public InferenceResponse createCompletion(InferenceRequest request) {
                        return null;
                    }

                    @Override
                    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
                        return null;
                    }

                    @Override
                    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
                        return null;
                    }

                    @Override
                    public String submitAsyncJob(InferenceRequest request) {
                        return null;
                    }

                    @Override
                    public AsyncJobStatus getJobStatus(String jobId) {
                        return null;
                    }

                    @Override
                    public AsyncJobStatus waitForJob(String jobId, java.time.Duration maxWaitTime,
                            java.time.Duration pollInterval) {
                        return null;
                    }

                    @Override
                    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) {
                        return null;
                    }

                    @Override
                    public List<ProviderInfo> listAvailableProviders() {
                        return List.of();
                    }

                    @Override
                    public ProviderInfo getProviderInfo(String providerId) {
                        return null;
                    }

                    @Override
                    public void setPreferredProvider(String providerId) {
                    }

                    @Override
                    public java.util.Optional<String> getPreferredProvider() {
                        return java.util.Optional.empty();
                    }

                    @Override
                    public List<ModelInfo> listModels() {
                        return List.of();
                    }

                    @Override
                    public List<ModelInfo> listModels(int offset, int limit) {
                        return List.of();
                    }

                    @Override
                    public java.util.Optional<ModelInfo> getModelInfo(String modelId) {
                        return java.util.Optional.empty();
                    }

                    @Override
                    public void pullModel(String modelSpec,
                            java.util.function.Consumer<PullProgress> progressCallback) {
                    }

                    @Override
                    public void deleteModel(String modelId) {
                    }

                    @Override
                    public McpRegistryManager mcpRegistry() {
                        return registry;
                    }
                };
            }
            return localClient;
        }
    }

    @Command(name = "add", description = "Add or update MCP server config from JSON")
    public static class Add extends McpSubcommand implements Runnable {
        @Parameters(index = "0", arity = "0..1", paramLabel = "<json>", description = "JSON payload containing mcpServers")
        public String inlineJson;

        @Option(names = { "--file" }, description = "Path to JSON file containing mcpServers")
        public String filePath;

        @Option(names = { "--from-url" }, description = "HTTP(S) URL returning JSON payload containing mcpServers")
        public String fromUrl;

        @Option(names = {
                "--from-registry" }, description = "Registry page URL or slug (e.g. qpd-v/mcp-image-downloader)")
        public String fromRegistry;

        @Option(names = {
                "--server" }, description = "Import only one server name from payload (when multiple mcpServers exist)")
        public String server;

        @Option(names = {
                "--list-from-registry" }, description = "List server names from source payload without saving")
        public boolean listFromRegistry;

        @Option(names = { "--name" }, description = "MCP server name for structured add")
        public String name;

        @Option(names = { "--transport" }, description = "Transport type for structured add (stdio|http|websocket)")
        String transport;

        @Option(names = { "--command" }, description = "Command for stdio transport (structured add)")
        String command;

        @Option(names = { "--url" }, description = "URL for http/websocket transport (structured add)")
        String url;

        @Option(names = { "--args-json" }, description = "JSON array string for args (structured add)")
        public String argsJson;

        @Option(names = { "--env-json" }, description = "JSON object string for env (structured add)")
        public String envJson;

        @Option(names = { "--enabled" }, description = "Enabled flag for structured add")
        public Boolean enabled;

        @Override
        public void run() {
            try {
                if (server != null && !server.isBlank() && name != null && !name.isBlank()) {
                    throw new IllegalArgumentException(
                            "Do not combine --name (structured add) with other add inputs.");
                }

                if (fromUrl != null && !fromUrl.isBlank() && (name != null || transport != null || command != null
                        || argsJson != null || envJson != null)) {
                    throw new IllegalArgumentException(
                            "Do not combine --from-url with other add inputs.");
                }

                if (listFromRegistry) {
                    if (!hasAnySource()) {
                        throw new IllegalArgumentException(
                                "Use --list-from-registry with one source: <json>/--file, --from-url, or --from-registry.");
                    }
                    if (name != null && !name.isBlank()) {
                        throw new IllegalArgumentException(
                                "--name is for structured add; use --server to filter payload sources.");
                    }
                    if (transport != null || command != null || url != null || argsJson != null || envJson != null
                            || enabled != null) {
                        throw new IllegalArgumentException(
                                "--list-from-registry cannot be combined with structured add flags.");
                    }
                    String selector = (server != null && !server.isBlank()) ? server.trim() : null;
                    List<String> discovered = client().mcpRegistry().add(new McpAddRequest(
                            inlineJson,
                            filePath,
                            fromUrl,
                            fromRegistry,
                            selector,
                            LIST_ONLY_MARKER,
                            null,
                            null,
                            null,
                            null,
                            null));
                    if (discovered.isEmpty()) {
                        System.out.println("No MCP servers discovered in source payload.");
                    } else {
                        System.out.println("Discovered MCP servers:");
                        discovered.forEach(entry -> System.out.printf("- %s%n", entry));
                    }
                    return;
                }

                String requestName = (server != null && !server.isBlank()) ? server.trim() : name;

                List<String> upserted = client().mcpRegistry().add(new McpAddRequest(
                        inlineJson,
                        filePath,
                        fromUrl,
                        fromRegistry,
                        requestName,
                        transport,
                        command,
                        url,
                        argsJson,
                        envJson,
                        enabled));

                System.out.printf("Saved %d MCP server(s): %s%n", upserted.size(), String.join(", ", upserted));
                System.out.printf("Registry: %s%n", client().mcpRegistry().registryPath());
            } catch (Exception e) {
                System.err.println("Failed to add MCP config: " + e.getMessage());
            }
        }

        private boolean hasAnySource() {
            return (inlineJson != null && !inlineJson.isBlank())
                    || (filePath != null && !filePath.isBlank())
                    || (fromUrl != null && !fromUrl.isBlank())
                    || (fromRegistry != null && !fromRegistry.isBlank());
        }

        private boolean isStructuredAdd(Add add) {
            return add.name != null && !add.name.isBlank();
        }
    }

    @Command(name = "show", description = "Show one MCP server config from registry")
    public static class ShowServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", paramLabel = "<name>", description = "MCP server name")
        String name;

        @Option(names = { "--json" }, description = "Show raw JSON")
        boolean json;

        @Override
        public void run() {
            try {
                var server = client().mcpRegistry().show(name);

                if (json) {
                    System.out.println(server.rawJson());
                    return;
                }

                System.out.printf("name: %s%n", server.name());
                System.out.printf("enabled: %s%n", server.enabled());
                System.out.printf("transport: %s%n", server.transport());
                System.out.printf("command: %s%n", server.command());
                System.out.printf("args: %d%n", server.argsCount());
                System.out.printf("env keys: %d%n", server.envKeys());
                System.out.printf("url: %s%n", server.url());
            } catch (Exception e) {
                System.err.println("Failed to show MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "import", description = "Import MCP server config from JSON file")
    public static class ImportServers extends McpSubcommand implements Runnable {
        @Option(names = { "--file" }, required = true, description = "Path to JSON file containing mcpServers")
        String filePath;

        @Option(names = { "--merge" }, description = "Merge imported servers with existing registry (default)")
        boolean merge = true;

        @Option(names = { "--replace" }, description = "Replace existing registry with imported servers")
        boolean replace;

        @Override
        public void run() {
            try {
                if (replace && merge) {
                    merge = false;
                }

                int imported = client().mcpRegistry().importFromFile(filePath, replace);
                int total = client().mcpRegistry().list().size();
                System.out.printf("Imported %d server(s) from %s%n", imported, filePath);
                System.out.printf("Registry now has %d server(s)%n", total);
                System.out.printf("Registry: %s%n", client().mcpRegistry().registryPath());
            } catch (Exception e) {
                System.err.println("Failed to import MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "export", description = "Export MCP server config to JSON file")
    public static class ExportServers extends McpSubcommand implements Runnable {
        @Option(names = { "--file" }, required = true, description = "Destination JSON file path")
        public String filePath;

        @Option(names = { "--name" }, description = "Export only one server name")
        public String name;

        @Option(names = { "-f", "--format" }, description = "Output format: table, json", defaultValue = "table")
        public String format;

        @Override
        public void run() {
            try {
                int exported = client().mcpRegistry().exportToFile(filePath, name);
                System.out.printf("Exported %d server(s) to %s%n", exported, filePath);
            } catch (Exception e) {
                System.err.println("Failed to export MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "remove", description = "Remove MCP server from registry")
    public static class Remove extends McpSubcommand implements Runnable {
        @Parameters(index = "0", paramLabel = "<name>", description = "MCP server name")
        public String name;

        @Override
        public void run() {
            try {
                client().mcpRegistry().remove(name);
                System.out.printf("Removed MCP server: %s%n", name);
                System.out.printf("Registry: %s%n", client().mcpRegistry().registryPath());
            } catch (Exception e) {
                System.err.println("Failed to remove MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "rename", description = "Rename MCP server in registry")
    public static class RenameServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", description = "Current MCP server name")
        public String oldName;

        @Parameters(index = "1", description = "New MCP server name")
        public String newName;

        @Override
        public void run() {
            try {
                client().mcpRegistry().rename(oldName, newName);
                System.out.printf("Renamed MCP server '%s' -> '%s'%n", oldName, newName);
            } catch (Exception e) {
                System.err.println("Failed to rename MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "edit", description = "Edit MCP server fields in registry")
    public static class EditServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", paramLabel = "<name>", description = "MCP server name")
        public String name;

        @Option(names = { "--transport" }, description = "New transport type (stdio|http|websocket)")
        public String transport;

        @Option(names = { "--command" }, description = "New executable command")
        public String command;

        @Option(names = { "--url" }, description = "New URL for http/websocket transport")
        public String url;

        @Option(names = { "--args-json" }, description = "JSON array string for args, e.g. '[\"a\",\"b\"]'")
        public String argsJson;

        @Option(names = { "--clear-args" }, description = "Remove args field")
        public boolean clearArgs;

        @Option(names = { "--env-json" }, description = "JSON object string for env, e.g. '{\"KEY\":\"VALUE\"}'")
        public String envJson;

        @Option(names = { "--clear-env" }, description = "Remove env field")
        public boolean clearEnv;

        @Option(names = { "--enabled" }, description = "Enabled flag (true/false)")
        public Boolean enabled;

        @Override
        public void run() {
            try {
                if (clearArgs && argsJson != null) {
                    throw new IllegalArgumentException("Use either --args-json or --clear-args, not both.");
                }
                if (clearEnv && envJson != null) {
                    throw new IllegalArgumentException("Use either --env-json or --clear-env, not both.");
                }

                client().mcpRegistry().edit(new McpEditRequest(
                        name, transport, command, url, argsJson, clearArgs, envJson, clearEnv, enabled));

                System.out.printf("Updated MCP server: %s%n", name);
            } catch (Exception e) {
                System.err.println("Failed to edit MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "enable", description = "Enable MCP server in registry")
    public static class EnableServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", paramLabel = "<name>", description = "MCP server name")
        public String name;

        @Override
        public void run() {
            try {
                client().mcpRegistry().setEnabled(name, true);
                System.out.printf("Enabled MCP server: %s%n", name);
            } catch (Exception e) {
                System.err.println("Failed to update MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "disable", description = "Disable MCP server in registry")
    public static class DisableServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", paramLabel = "<name>", description = "MCP server name")
        public String name;

        @Override
        public void run() {
            try {
                client().mcpRegistry().setEnabled(name, false);
                System.out.printf("Disabled MCP server: %s%n", name);
            } catch (Exception e) {
                System.err.println("Failed to update MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "list", description = "List MCP servers in registry")
    public static class ListServers extends McpSubcommand implements Runnable {
        @Override
        public void run() {
            try {
                var servers = client().mcpRegistry().list();
                if (servers.isEmpty()) {
                    System.out.printf("No MCP servers configured. Registry: %s%n",
                            client().mcpRegistry().registryPath());
                    return;
                }

                System.out.println("MCP servers:");
                servers.forEach(entry -> System.out.printf("- %s (%s)%n", entry.name(),
                        entry.enabled() ? "enabled" : "disabled"));
                System.out.printf("Registry: %s%n", client().mcpRegistry().registryPath());
            } catch (Exception e) {
                System.err.println("Failed to list MCP config: " + e.getMessage());
            }
        }
    }

    @Command(name = "doctor", description = "Validate MCP server registry and print per-server diagnostics")
    public static class DoctorServers extends McpSubcommand implements Runnable {
        @Override
        public void run() {
            try {
                McpDoctorReport report = client().mcpRegistry().doctor();
                if (report.total() == 0) {
                    System.out.println("No MCP servers configured.");
                    System.out.printf("Registry: %s%n", report.registryPath());
                    return;
                }
                report.entries().forEach(entry -> {
                    if (entry.errors().isEmpty()) {
                        System.out.printf("[OK] %s%n", entry.name());
                    } else {
                        System.out.printf("[ERROR] %s%n", entry.name());
                        entry.errors().forEach(err -> System.out.printf("  - %s%n", err));
                    }
                });
                System.out.printf("MCP doctor summary: passed=%d failed=%d total=%d%n",
                        report.passed(), report.failed(), report.total());
                System.out.printf("Registry: %s%n", report.registryPath());
            } catch (Exception e) {
                System.err.println("Failed to run MCP doctor: " + e.getMessage());
            }
        }
    }

    @Command(name = "test", description = "Test an MCP server from registry with initialize + discovery")
    public static class TestServer extends McpSubcommand implements Runnable {
        @Parameters(index = "0", arity = "0..1", paramLabel = "<name>", description = "MCP server name")
        public String name;

        @Option(names = { "--all" }, description = "Test all enabled MCP servers from registry")
        public boolean all;

        @Option(names = {
                "--timeout-ms" }, description = "Timeout in milliseconds per RPC request", defaultValue = "8000")
        public long timeoutMs;

        @Override
        public void run() {
            try {
                if (name == null && !all) {
                    System.err.println("Provide <name> or use --all.");
                    return;
                }
                McpTestReport report = client().mcpRegistry().test(name, all, timeoutMs);
                for (McpTestEntry entry : report.entries()) {
                    if (entry.success()) {
                        System.out.printf("MCP test passed for '%s'%n", entry.name());
                        System.out.printf("- tools: %d%n", entry.tools());
                        System.out.printf("- resources: %d%n", entry.resources());
                        System.out.printf("- prompts: %d%n", entry.prompts());
                    } else {
                        System.err.printf("MCP test failed for '%s': %s%n", entry.name(), entry.error());
                    }
                }
                if (all || report.total() > 1) {
                    System.out.printf("MCP test summary: passed=%d failed=%d total=%d%n",
                            report.passed(), report.failed(), report.total());
                }
            } catch (Exception e) {
                System.err.println("MCP test failed: " + e.getMessage());
            }
        }
    }

    public static JsonObject parseJsonObject(String json) {
        try (JsonReader reader = Json.createReader(new StringReader(json))) {
            return reader.readObject();
        }
    }

    public static JsonObject loadRegistry() {
        try {
            Path path = Path.of(System.getProperty("user.home"), ".gollek", "mcp", "servers.json");
            if (Files.exists(path)) {
                return parseJsonObject(Files.readString(path));
            }
        } catch (Exception e) {
            // ignore
        }
        return Json.createObjectBuilder()
                .add("mcpServers", Json.createObjectBuilder().build())
                .build();
    }

    public static List<String> validateServers(JsonObject servers) {
        List<String> errors = new ArrayList<>();
        if (servers == null) {
            errors.add("JSON is null");
            return errors;
        }
        JsonObject mcpServers = servers.getJsonObject("mcpServers");
        JsonObject toValidate = (mcpServers != null) ? mcpServers : servers;

        for (Map.Entry<String, JsonValue> entry : toValidate.entrySet()) {
            if (!(entry.getValue() instanceof JsonObject)) {
                errors.add("Server '" + entry.getKey() + "' is not a JSON object");
                continue;
            }
            JsonObject config = (JsonObject) entry.getValue();
            if (!config.containsKey("command")) {
                errors.add("requires non-empty 'command'");
            }
        }
        return errors;
    }
}
