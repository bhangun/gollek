package tech.kayys.gollek.mcp.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.json.Json;
import jakarta.json.JsonArray;
import jakarta.json.JsonObject;
import jakarta.json.JsonObjectBuilder;
import jakarta.json.JsonReader;
import jakarta.json.JsonString;
import jakarta.json.JsonValue;
import jakarta.json.stream.JsonGenerator;
import tech.kayys.gollek.sdk.core.exception.SdkException;
import tech.kayys.gollek.sdk.core.mcp.McpAddRequest;
import tech.kayys.gollek.sdk.core.mcp.McpDoctorEntry;
import tech.kayys.gollek.sdk.core.mcp.McpDoctorReport;
import tech.kayys.gollek.sdk.core.mcp.McpEditRequest;
import tech.kayys.gollek.sdk.core.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.core.mcp.McpServerSummary;
import tech.kayys.gollek.sdk.core.mcp.McpServerView;
import tech.kayys.gollek.sdk.core.mcp.McpTestEntry;
import tech.kayys.gollek.sdk.core.mcp.McpTestReport;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@ApplicationScoped
public class McpRegistryEngine implements McpRegistryManager {

    public static final String MCP_SERVERS_KEY = "mcpServers";
    private static final String LIST_ONLY_MARKER = "__MCP_LIST_ONLY__";
    private static final String REGISTRY_MODE_PROP = "gollek.mcp.registry.mode";
    private static final String REGISTRY_MODE_ENV = "GOLLEK_MCP_REGISTRY_MODE";
    private static final String REGISTRY_API_BASE_URL_PROP = "gollek.mcp.registry.api.base-url";
    private static final String REGISTRY_API_BASE_URL_ENV = "GOLLEK_MCP_REGISTRY_API_BASE_URL";
    private static final String REGISTRY_API_TOKEN_PROP = "gollek.mcp.registry.api-token";
    private static final String REGISTRY_API_TOKEN_ENV = "GOLLEK_MCP_REGISTRY_API_TOKEN";
    private static final String ENTERPRISE_ENABLED_PROP = "gollek.enterprise.enabled";
    private static final String ENTERPRISE_ENABLED_ENV = "GOLLEK_ENTERPRISE_ENABLED";
    private static final String EDITION_PROP = "gollek.edition";
    private static final String EDITION_ENV = "GOLLEK_EDITION";
    private static final String MULTITENANCY_PROP = "wayang.multitenancy.enabled";
    private static final String MULTITENANCY_ENV = "WAYANG_MULTITENANCY_ENABLED";
    private static final String TENANT_HEADER_ENABLED_PROP = "gollek.mcp.registry.tenant-header.enabled";
    private static final String TENANT_HEADER_ENABLED_ENV = "GOLLEK_MCP_REGISTRY_TENANT_HEADER_ENABLED";
    private static final String TENANT_HEADER_NAME_PROP = "gollek.mcp.registry.tenant-header.name";
    private static final String TENANT_HEADER_NAME_ENV = "GOLLEK_MCP_REGISTRY_TENANT_HEADER_NAME";
    private static final String TENANT_ID_PROP = "gollek.tenant.id";
    private static final String TENANT_ID_ENV = "GOLLEK_TENANT_ID";
    private static final String REGISTRY_API_PREFIX = "/api/v1/mcp/registry";
    private static final String REMOTE_REGISTRY_MODE = "remote";
    private static final String ENTERPRISE_EDITION = "enterprise";

    @Override
    public String registryPath() {
        if (useRemoteRegistry()) {
            return remoteEndpoint("/servers");
        }
        return path().toAbsolutePath().toString();
    }

    @Override
    public List<String> add(McpAddRequest request) throws SdkException {
        try {
            boolean listOnly = LIST_ONLY_MARKER.equals(request.transport());
            String json = resolveJsonOrBuildStructured(request);
            if (json == null) {
                throw new SdkException("SDK_ERR_MCP_ADD", "Provide JSON payload, --file, or structured add flags.");
            }
            JsonObject parsed = parseJsonObject(json);
            JsonObject incomingServers = canonicalizeServers(extractServersObject(parsed));
            if (hasText(request.name()) && !isStructuredAdd(request)) {
                incomingServers = selectServer(incomingServers, request.name().trim());
            }
            if (incomingServers.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_ADD", "No servers found. Expected key `mcpServers`.");
            }

            if (listOnly) {
                return incomingServers.keySet().stream().sorted().toList();
            }

            if (useRemoteRegistry()) {
                if (isStructuredAdd(request)) {
                    upsertRemoteServer(request.name(), incomingServers.getJsonObject(request.name()));
                    return List.of(request.name());
                }
                JsonObject response = postJson(remoteEndpoint("/import"), Json.createObjectBuilder()
                        .add("sourceType", "RAW")
                        .add("source", json)
                        .add("serverName", request.name() == null ? JsonValue.NULL : Json.createValue(request.name()))
                        .build());
                JsonArray names = response.getJsonArray("importedNames");
                if (names == null) {
                    return incomingServers.keySet().stream().sorted().toList();
                }
                List<String> out = new ArrayList<>();
                for (JsonValue value : names) {
                    if (value instanceof JsonString jsonString) {
                        out.add(jsonString.getString());
                    }
                }
                return out.stream().sorted().toList();
            }

            List<String> validationErrors = validateServers(incomingServers);
            if (!validationErrors.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_ADD", "Invalid MCP config: " + String.join("; ", validationErrors));
            }

            JsonObject existingRoot = loadRegistry();
            JsonObject existingServers = extractServersObject(existingRoot);
            JsonObjectBuilder mergedServersBuilder = Json.createObjectBuilder();
            existingServers.forEach(mergedServersBuilder::add);

            List<String> upserted = new ArrayList<>();
            incomingServers.forEach((name, config) -> {
                mergedServersBuilder.add(name, config);
                upserted.add(name);
            });

            JsonObject mergedRoot = Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, mergedServersBuilder.build())
                    .build();
            saveRegistry(mergedRoot);
            return upserted;
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_ADD", withCause("Failed to add MCP config", e), e);
        }
    }

    @Override
    public McpServerView show(String name) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                JsonObject server = getRemoteServer(name);
                String transport = stringValue(server.get("transport"), "stdio");
                boolean enabled = !server.containsKey("enabled") || server.getBoolean("enabled", true);
                String command = stringValue(server.get("command"), null);
                String url = stringValue(server.get("url"), null);
                JsonArray args = parseJsonArrayField(server.get("argsJson"));
                JsonObject env = parseJsonObjectField(server.get("envJson"));
                JsonObjectBuilder rawBuilder = Json.createObjectBuilder();
                if (transport != null) {
                    rawBuilder.add("transport", transport);
                }
                if (command != null) {
                    rawBuilder.add("command", command);
                }
                if (url != null) {
                    rawBuilder.add("url", url);
                }
                if (args != null) {
                    rawBuilder.add("args", args);
                }
                if (env != null) {
                    rawBuilder.add("env", env);
                }
                rawBuilder.add("enabled", enabled);
                JsonObject raw = rawBuilder.build();
                return new McpServerView(
                        name,
                        enabled,
                        transport != null ? transport : "stdio",
                        command != null ? command : "<none>",
                        args != null ? args.size() : 0,
                        env != null ? env.size() : 0,
                        url != null ? url : "<none>",
                        raw.toString());
            }
            JsonObject servers = extractServersObject(loadRegistry());
            JsonObject server = servers.getJsonObject(name);
            if (server == null) {
                throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
            }
            String transport = resolveTransport(server);
            boolean enabled = !server.containsKey("enabled") || server.getBoolean("enabled", true);
            JsonArray args = server.getJsonArray("args");
            JsonObject env = server.getJsonObject("env");
            return new McpServerView(
                    name,
                    enabled,
                    transport,
                    resolveCommand(server) != null ? resolveCommand(server) : "<none>",
                    args != null ? args.size() : 0,
                    env != null ? env.size() : 0,
                    resolveUrl(server) != null ? resolveUrl(server) : "<none>",
                    server.toString());
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_SHOW", withCause("Failed to show MCP config", e), e);
        }
    }

    @Override
    public List<McpServerSummary> list() throws SdkException {
        try {
            if (useRemoteRegistry()) {
                JsonArray servers = getJsonArray(remoteEndpoint("/servers"));
                List<McpServerSummary> list = new ArrayList<>();
                for (JsonValue value : servers) {
                    if (!(value instanceof JsonObject server)) {
                        continue;
                    }
                    String name = stringValue(server.get("name"), null);
                    if (name == null || name.isBlank()) {
                        continue;
                    }
                    boolean enabled = !server.containsKey("enabled") || server.getBoolean("enabled", true);
                    list.add(new McpServerSummary(name, enabled));
                }
                list.sort(Comparator.comparing(McpServerSummary::name));
                return list;
            }
            JsonObject servers = extractServersObject(loadRegistry());
            return servers.entrySet().stream()
                    .sorted(Comparator.comparing(Map.Entry::getKey))
                    .map(entry -> {
                        JsonObject server = entry.getValue().asJsonObject();
                        boolean enabled = !server.containsKey("enabled") || server.getBoolean("enabled", true);
                        return new McpServerSummary(entry.getKey(), enabled);
                    })
                    .toList();
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_LIST", withCause("Failed to list MCP config", e), e);
        }
    }

    @Override
    public void remove(String name) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                int status = deleteNoContent(remoteEndpoint("/servers/" + urlEncode(name)));
                if (status == 404) {
                    throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
                }
                if (status < 200 || status >= 300) {
                    throw new SdkException("SDK_ERR_MCP_REMOVE", "Registry API returned HTTP " + status);
                }
                return;
            }
            JsonObject servers = extractServersObject(loadRegistry());
            if (!servers.containsKey(name)) {
                throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
            }
            JsonObjectBuilder remaining = Json.createObjectBuilder();
            servers.forEach((key, value) -> {
                if (!key.equals(name)) {
                    remaining.add(key, value);
                }
            });
            saveRegistry(Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, remaining.build())
                    .build());
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_REMOVE", withCause("Failed to remove MCP config", e), e);
        }
    }

    @Override
    public void rename(String oldName, String newName) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                McpServerView view = show(oldName);
                JsonObject server = parseJsonObject(view.rawJson());
                upsertRemoteServer(newName, server);
                remove(oldName);
                return;
            }
            JsonObject servers = extractServersObject(loadRegistry());
            JsonObject existing = servers.getJsonObject(oldName);
            if (existing == null) {
                throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + oldName);
            }
            if (servers.containsKey(newName)) {
                throw new SdkException("SDK_ERR_MCP_RENAME", "Server already exists: " + newName);
            }
            JsonObjectBuilder updated = Json.createObjectBuilder();
            servers.forEach((key, value) -> {
                if (key.equals(oldName)) {
                    updated.add(newName, value);
                } else {
                    updated.add(key, value);
                }
            });
            saveRegistry(Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, updated.build())
                    .build());
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_RENAME", withCause("Failed to rename MCP config", e), e);
        }
    }

    @Override
    public void edit(McpEditRequest request) throws SdkException {
        try {
            if (request.clearArgs() && request.argsJson() != null) {
                throw new SdkException("SDK_ERR_MCP_EDIT", "Use either argsJson or clearArgs, not both.");
            }
            if (request.clearEnv() && request.envJson() != null) {
                throw new SdkException("SDK_ERR_MCP_EDIT", "Use either envJson or clearEnv, not both.");
            }

            JsonObject servers = null;
            JsonObject existing;
            if (useRemoteRegistry()) {
                existing = parseJsonObject(show(request.name()).rawJson());
            } else {
                servers = extractServersObject(loadRegistry());
                existing = servers.getJsonObject(request.name());
                if (existing == null) {
                    throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + request.name());
                }
            }

            Map<String, JsonValue> fields = new LinkedHashMap<>();
            existing.forEach(fields::put);
            if (request.transport() != null) {
                fields.put("transport", Json.createValue(request.transport()));
            }
            if (request.command() != null) {
                fields.put("command", Json.createValue(request.command()));
            }
            if (request.url() != null) {
                fields.put("url", Json.createValue(request.url()));
            }
            if (request.enabled() != null) {
                fields.put("enabled", request.enabled() ? JsonValue.TRUE : JsonValue.FALSE);
            }
            if (request.clearArgs()) {
                fields.remove("args");
            } else if (request.argsJson() != null) {
                fields.put("args", parseJsonArray(request.argsJson()));
            }
            if (request.clearEnv()) {
                fields.remove("env");
            } else if (request.envJson() != null) {
                fields.put("env", parseJsonObject(request.envJson()));
            }

            JsonObjectBuilder updatedServerBuilder = Json.createObjectBuilder();
            fields.forEach(updatedServerBuilder::add);
            JsonObject updatedServer = canonicalizeServer(updatedServerBuilder.build());

            JsonObject validationProbe = Json.createObjectBuilder()
                    .add(request.name(), updatedServer)
                    .build();
            List<String> validationErrors = validateServers(validationProbe);
            if (!validationErrors.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_EDIT", "Invalid MCP config: " + String.join("; ", validationErrors));
            }

            if (useRemoteRegistry()) {
                upsertRemoteServer(request.name(), updatedServer);
            } else {
                JsonObjectBuilder updatedServers = Json.createObjectBuilder();
                servers.forEach((key, value) -> {
                    if (key.equals(request.name())) {
                        updatedServers.add(key, updatedServer);
                    } else {
                        updatedServers.add(key, value);
                    }
                });
                saveRegistry(Json.createObjectBuilder()
                        .add(MCP_SERVERS_KEY, updatedServers.build())
                        .build());
            }
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_EDIT", withCause("Failed to edit MCP config", e), e);
        }
    }

    @Override
    public void setEnabled(String name, boolean enabled) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                JsonObject server = parseJsonObject(show(name).rawJson());
                JsonObjectBuilder updated = Json.createObjectBuilder();
                server.forEach(updated::add);
                updated.add("enabled", enabled);
                upsertRemoteServer(name, updated.build());
                return;
            }
            JsonObject servers = extractServersObject(loadRegistry());
            JsonObject server = servers.getJsonObject(name);
            if (server == null) {
                throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
            }
            JsonObjectBuilder updatedServer = Json.createObjectBuilder();
            server.forEach(updatedServer::add);
            updatedServer.add("enabled", enabled);
            JsonObjectBuilder updatedServers = Json.createObjectBuilder();
            servers.forEach((key, value) -> {
                if (key.equals(name)) {
                    updatedServers.add(key, updatedServer.build());
                } else {
                    updatedServers.add(key, value);
                }
            });
            saveRegistry(Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, updatedServers.build())
                    .build());
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_ENABLE", withCause("Failed to update MCP config", e), e);
        }
    }

    @Override
    public int importFromFile(String filePath, boolean replace) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                JsonObject importedRoot = parseJsonObject(Files.readString(Path.of(filePath), StandardCharsets.UTF_8));
                JsonObject importedServers = canonicalizeServers(extractServersObject(importedRoot));
                if (replace) {
                    for (McpServerSummary server : list()) {
                        remove(server.name());
                    }
                }
                JsonObject response = postJson(remoteEndpoint("/import"), Json.createObjectBuilder()
                        .add("sourceType", "RAW")
                        .add("source", Json.createObjectBuilder().add(MCP_SERVERS_KEY, importedServers).build().toString())
                        .build());
                return response.getInt("importedCount", importedServers.size());
            }
            JsonObject importedRoot = parseJsonObject(Files.readString(Path.of(filePath), StandardCharsets.UTF_8));
            JsonObject importedServers = canonicalizeServers(extractServersObject(importedRoot));
            if (importedServers.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_IMPORT", "No servers found in import file.");
            }
            List<String> validationErrors = validateServers(importedServers);
            if (!validationErrors.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_IMPORT", "Invalid MCP config: " + String.join("; ", validationErrors));
            }

            JsonObject finalServers;
            if (replace) {
                finalServers = importedServers;
            } else {
                JsonObject existing = extractServersObject(loadRegistry());
                JsonObjectBuilder merged = Json.createObjectBuilder();
                existing.forEach(merged::add);
                importedServers.forEach(merged::add);
                finalServers = merged.build();
            }
            saveRegistry(Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, finalServers)
                    .build());
            return importedServers.size();
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_IMPORT", withCause("Failed to import MCP config", e), e);
        }
    }

    @Override
    public int exportToFile(String filePath, String name) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                JsonObjectBuilder serversBuilder = Json.createObjectBuilder();
                if (name != null && !name.isBlank()) {
                    serversBuilder.add(name, parseJsonObject(show(name).rawJson()));
                } else {
                    for (McpServerSummary summary : list()) {
                        serversBuilder.add(summary.name(), parseJsonObject(show(summary.name()).rawJson()));
                    }
                }
                JsonObject toExport = Json.createObjectBuilder()
                        .add(MCP_SERVERS_KEY, serversBuilder.build())
                        .build();
                Path out = Path.of(filePath);
                if (out.getParent() != null) {
                    Files.createDirectories(out.getParent());
                }
                StringWriter writer = new StringWriter();
                var factory = Json.createWriterFactory(Map.of(JsonGenerator.PRETTY_PRINTING, true));
                factory.createWriter(writer).writeObject(toExport);
                Files.writeString(out, writer.toString(), StandardCharsets.UTF_8);
                return extractServersObject(toExport).size();
            }
            JsonObject servers = extractServersObject(loadRegistry());
            if (servers.isEmpty()) {
                throw new SdkException("SDK_ERR_MCP_EXPORT", "No MCP servers configured to export.");
            }
            JsonObject toExport;
            if (name != null && !name.isBlank()) {
                JsonObject server = servers.getJsonObject(name);
                if (server == null) {
                    throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
                }
                toExport = Json.createObjectBuilder()
                        .add(MCP_SERVERS_KEY, Json.createObjectBuilder().add(name, server).build())
                        .build();
            } else {
                toExport = Json.createObjectBuilder().add(MCP_SERVERS_KEY, servers).build();
            }

            Path out = Path.of(filePath);
            if (out.getParent() != null) {
                Files.createDirectories(out.getParent());
            }
            StringWriter writer = new StringWriter();
            var factory = Json.createWriterFactory(Map.of(JsonGenerator.PRETTY_PRINTING, true));
            factory.createWriter(writer).writeObject(toExport);
            Files.writeString(out, writer.toString(), StandardCharsets.UTF_8);
            return extractServersObject(toExport).size();
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_EXPORT", withCause("Failed to export MCP config", e), e);
        }
    }

    @Override
    public McpDoctorReport doctor() throws SdkException {
        try {
            if (useRemoteRegistry()) {
                List<McpDoctorEntry> entries = new ArrayList<>();
                int passed = 0;
                int failed = 0;
                for (McpServerSummary summary : list()) {
                    JsonObject probe = Json.createObjectBuilder()
                            .add(summary.name(), parseJsonObject(show(summary.name()).rawJson()))
                            .build();
                    List<String> errors = validateServers(probe);
                    entries.add(new McpDoctorEntry(summary.name(), errors));
                    if (errors.isEmpty()) {
                        passed++;
                    } else {
                        failed++;
                    }
                }
                return new McpDoctorReport(entries, passed, failed, entries.size(), registryPath());
            }
            JsonObject servers = extractServersObject(loadRegistry());
            List<McpDoctorEntry> entries = new ArrayList<>();
            int passed = 0;
            int failed = 0;
            for (Map.Entry<String, JsonValue> entry : servers.entrySet()) {
                String name = entry.getKey();
                JsonObject probe = Json.createObjectBuilder().add(name, entry.getValue()).build();
                List<String> errors = validateServers(probe);
                entries.add(new McpDoctorEntry(name, errors));
                if (errors.isEmpty()) {
                    passed++;
                } else {
                    failed++;
                }
            }
            return new McpDoctorReport(entries, passed, failed, servers.size(), registryPath());
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_DOCTOR", withCause("Failed to run MCP doctor", e), e);
        }
    }

    @Override
    public McpTestReport test(String name, boolean all, long timeoutMs) throws SdkException {
        try {
            if (useRemoteRegistry()) {
                throw new SdkException("SDK_ERR_MCP_TEST",
                        "Remote registry mode does not support local process test. Switch to local registry mode for `mcp test`.");
            }
            JsonObject servers = extractServersObject(loadRegistry());
            List<String> targets;
            if (all) {
                targets = servers.entrySet().stream()
                        .filter(entry -> {
                            JsonObject server = entry.getValue().asJsonObject();
                            return !server.containsKey("enabled") || server.getBoolean("enabled", true);
                        })
                        .map(Map.Entry::getKey)
                        .sorted()
                        .toList();
            } else {
                if (name == null || name.isBlank()) {
                    throw new SdkException("SDK_ERR_MCP_TEST", "Provide a server name or use all=true.");
                }
                targets = List.of(name);
            }

            List<McpTestEntry> entries = new ArrayList<>();
            int passed = 0;
            int failed = 0;
            for (String target : targets) {
                JsonObject server = servers.getJsonObject(target);
                if (server == null) {
                    entries.add(new McpTestEntry(target, false, 0, 0, 0, "Server not found in registry"));
                    failed++;
                    continue;
                }
                McpTestEntry entry = runSingleServerTest(target, server, timeoutMs);
                entries.add(entry);
                if (entry.success()) {
                    passed++;
                } else {
                    failed++;
                }
            }
            return new McpTestReport(entries, passed, failed, targets.size());
        } catch (SdkException e) {
            throw e;
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MCP_TEST", withCause("Failed to test MCP server(s)", e), e);
        }
    }

    private McpTestEntry runSingleServerTest(String serverName, JsonObject server, long timeoutMs) {
        Process process = null;
        try {
            String transport = resolveTransport(server);
            if (!"stdio".equals(transport)) {
                return new McpTestEntry(serverName, false, 0, 0, 0, "Unsupported transport: " + transport);
            }
            String command = resolveCommand(server);
            if (command == null || command.isBlank()) {
                return new McpTestEntry(serverName, false, 0, 0, 0, "Missing required field: command");
            }
            List<String> args = readStringArrayAsList(server.get("args"));
            List<String> cmd = new ArrayList<>();
            cmd.add(command);
            cmd.addAll(args);

            ProcessBuilder pb = new ProcessBuilder(cmd);
            JsonObject envObj = server.getJsonObject("env");
            if (envObj != null) {
                envObj.forEach((k, v) -> {
                    if (v != null && v.getValueType() != JsonValue.ValueType.NULL) {
                        pb.environment().put(k, jsonScalarToString(v));
                    }
                });
            }

            process = pb.start();
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
                 BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
                 BufferedReader errReader = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {

                long requestId = 1L;
                JsonObject initParams = Json.createObjectBuilder()
                        .add("protocolVersion", "2025-11-05")
                        .add("capabilities", Json.createObjectBuilder())
                        .add("clientInfo", Json.createObjectBuilder()
                                .add("name", "gollek-cli")
                                .add("version", "1.0.0"))
                        .build();

                JsonObject initialize = sendRequest(reader, writer, requestId++, "initialize", initParams, timeoutMs);
                if (initialize.containsKey("error")) {
                    return new McpTestEntry(serverName, false, 0, 0, 0, "initialize error: " + initialize.get("error"));
                }
                sendNotification(writer, "notifications/initialized", Json.createObjectBuilder().build());
                int tools = listCount(reader, writer, requestId++, "tools/list", "tools", timeoutMs);
                int resources = listCount(reader, writer, requestId++, "resources/list", "resources", timeoutMs);
                int prompts = listCount(reader, writer, requestId++, "prompts/list", "prompts", timeoutMs);
                return new McpTestEntry(serverName, true, tools, resources, prompts, null);
            }
        } catch (Exception e) {
            return new McpTestEntry(serverName, false, 0, 0, 0, e.getMessage());
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
    }

    private int listCount(
            BufferedReader reader,
            BufferedWriter writer,
            long id,
            String method,
            String resultField,
            long timeoutMillis) throws IOException {
        JsonObject response = sendRequest(reader, writer, id, method, null, timeoutMillis);
        if (response.containsKey("error")) {
            return 0;
        }
        JsonObject result = response.getJsonObject("result");
        if (result == null) {
            return 0;
        }
        JsonArray arr = result.getJsonArray(resultField);
        return arr != null ? arr.size() : 0;
    }

    private JsonObject sendRequest(
            BufferedReader reader,
            BufferedWriter writer,
            long id,
            String method,
            JsonObject params,
            long timeoutMillis) throws IOException {
        JsonObjectBuilder request = Json.createObjectBuilder()
                .add("jsonrpc", "2.0")
                .add("id", id)
                .add("method", method);
        if (params != null) {
            request.add("params", params);
        }
        writer.write(request.build().toString());
        writer.newLine();
        writer.flush();

        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (System.currentTimeMillis() < deadline) {
            if (!reader.ready()) {
                sleep(20);
                continue;
            }
            String line = reader.readLine();
            if (line == null || line.isBlank()) {
                continue;
            }
            try (JsonReader jsonReader = Json.createReader(new StringReader(line))) {
                JsonObject obj = jsonReader.readObject();
                JsonValue idValue = obj.get("id");
                if (idValue == null) {
                    continue;
                }
                if (idValue.getValueType() == JsonValue.ValueType.NUMBER
                        && obj.getJsonNumber("id").longValue() == id) {
                    return obj;
                }
                if (idValue.getValueType() == JsonValue.ValueType.STRING
                        && String.valueOf(id).equals(obj.getString("id", ""))) {
                    return obj;
                }
            } catch (Exception ignored) {
            }
        }
        throw new IOException("Timed out waiting for response to " + method);
    }

    private void sendNotification(BufferedWriter writer, String method, JsonObject params) throws IOException {
        JsonObjectBuilder notification = Json.createObjectBuilder()
                .add("jsonrpc", "2.0")
                .add("method", method);
        if (params != null) {
            notification.add("params", params);
        }
        writer.write(notification.build().toString());
        writer.newLine();
        writer.flush();
    }

    private static Path path() {
        return Path.of(System.getProperty("user.home"), ".gollek", "mcp", "servers.json");
    }

    private JsonObject loadRegistry() throws IOException {
        Path path = path();
        if (!Files.exists(path)) {
            return Json.createObjectBuilder()
                    .add(MCP_SERVERS_KEY, Json.createObjectBuilder().build())
                    .build();
        }
        String content = Files.readString(path, StandardCharsets.UTF_8);
        return normalizeRoot(parseJsonObject(content));
    }

    private void saveRegistry(JsonObject root) throws IOException {
        Path path = path();
        Files.createDirectories(path.getParent());
        StringWriter writer = new StringWriter();
        var factory = Json.createWriterFactory(Map.of(JsonGenerator.PRETTY_PRINTING, true));
        factory.createWriter(writer).writeObject(normalizeRoot(root));
        Files.writeString(path, writer.toString(), StandardCharsets.UTF_8);
    }

    private String resolveJsonOrBuildStructured(McpAddRequest request) throws IOException {
        boolean hasInlineOrFile = (request.inlineJson() != null && !request.inlineJson().isBlank())
                || (request.filePath() != null && !request.filePath().isBlank());
        boolean hasFromUrl = request.fromUrl() != null && !request.fromUrl().isBlank();
        boolean hasFromRegistry = request.fromRegistry() != null && !request.fromRegistry().isBlank();
        boolean hasStructured = isStructuredAdd(request);
        int sourceCount = (hasInlineOrFile ? 1 : 0) + (hasFromUrl ? 1 : 0) + (hasFromRegistry ? 1 : 0) + (hasStructured ? 1 : 0);

        if (sourceCount > 1) {
            throw new IllegalArgumentException("Use one add source only: <json>/--file, --from-url, --from-registry, or --name.");
        }
        if (hasFromUrl) {
            return fetchUrl(request.fromUrl());
        }
        if (hasFromRegistry) {
            return fetchRegistryConfig(request.fromRegistry());
        }
        if (hasInlineOrFile) {
            if (request.name() != null && !request.name().isBlank()) {
                throw new IllegalArgumentException("Do not combine JSON input (--file/<json>) with structured flags (--name ...).");
            }
            if (request.inlineJson() != null && !request.inlineJson().isBlank()) {
                return request.inlineJson();
            }
            if (request.filePath() != null && !request.filePath().isBlank()) {
                return Files.readString(Path.of(request.filePath()), StandardCharsets.UTF_8);
            }
        }
        if (request.name() == null || request.name().isBlank()) {
            return null;
        }

        JsonObjectBuilder server = Json.createObjectBuilder();
        if (request.transport() != null && !request.transport().isBlank()) {
            server.add("transport", request.transport());
        }
        if (request.command() != null && !request.command().isBlank()) {
            server.add("command", request.command());
        }
        if (request.url() != null && !request.url().isBlank()) {
            server.add("url", request.url());
        }
        if (request.argsJson() != null) {
            server.add("args", parseJsonArray(request.argsJson()));
        }
        if (request.envJson() != null) {
            server.add("env", parseJsonObject(request.envJson()));
        }
        if (request.enabled() != null) {
            server.add("enabled", request.enabled());
        }

        JsonObject payload = Json.createObjectBuilder()
                .add(MCP_SERVERS_KEY, Json.createObjectBuilder()
                        .add(request.name(), server.build())
                        .build())
                .build();
        return payload.toString();
    }

    private boolean isStructuredAdd(McpAddRequest request) {
        return hasText(request.name())
                && !hasText(request.inlineJson())
                && !hasText(request.filePath())
                && !hasText(request.fromUrl())
                && !hasText(request.fromRegistry());
    }

    private String fetchRegistryConfig(String source) {
        String resolved = normalizeRegistrySource(source);
        String body = fetchUrl(resolved);
        return extractMcpConfigJson(body, resolved);
    }

    private String normalizeRegistrySource(String source) {
        String trimmed = source == null ? "" : source.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("Registry source is empty.");
        }
        String lower = trimmed.toLowerCase();
        if (lower.startsWith("http://") || lower.startsWith("https://")) {
            return trimmed;
        }
        String slug = trimmed.startsWith("/") ? trimmed.substring(1) : trimmed;
        return "https://mcpservers.org/servers/" + slug;
    }

    private String extractMcpConfigJson(String content, String sourceHint) {
        if (content == null || content.isBlank()) {
            throw new IllegalArgumentException("Empty response from registry source: " + sourceHint);
        }

        String decoded = decodeHtmlJsonEntities(content);

        JsonObject direct = tryParseObject(decoded.trim());
        if (direct != null && !extractServersObject(direct).isEmpty()) {
            return direct.toString();
        }

        Matcher fenced = Pattern.compile("```(?:json)?\\s*([\\s\\S]*?)```", Pattern.CASE_INSENSITIVE).matcher(decoded);
        while (fenced.find()) {
            String candidate = fenced.group(1);
            JsonObject parsed = tryParseObject(candidate);
            if (parsed != null && !extractServersObject(parsed).isEmpty()) {
                return parsed.toString();
            }
        }

        int marker = decoded.indexOf("\"mcpServers\"");
        if (marker >= 0) {
            String object = extractEnclosingObject(decoded, marker);
            JsonObject parsed = tryParseObject(object);
            if (parsed != null && !extractServersObject(parsed).isEmpty()) {
                return parsed.toString();
            }
        }

        throw new IllegalArgumentException("Could not extract valid MCP config from registry source: " + sourceHint);
    }

    private JsonObject tryParseObject(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return null;
        }
        try {
            return parseJsonObject(candidate);
        } catch (Exception e) {
            return null;
        }
    }

    private String extractEnclosingObject(String text, int markerIndex) {
        int start = markerIndex;
        while (start >= 0 && text.charAt(start) != '{') {
            start--;
        }
        if (start < 0) {
            return "";
        }

        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int i = start; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }
            if (c == '"') {
                inString = true;
                continue;
            }
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    return text.substring(start, i + 1);
                }
            }
        }
        return "";
    }

    private String decodeHtmlJsonEntities(String raw) {
        return raw
                .replace("&quot;", "\"")
                .replace("&#34;", "\"")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">");
    }

    private String fetchUrl(String url) {
        try {
            HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Accept", "application/json")
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            int status = response.statusCode();
            if (status < 200 || status >= 300) {
                throw new IllegalArgumentException("HTTP " + status);
            }
            return response.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalArgumentException("Interrupted while fetching --from-url", e);
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to fetch --from-url: " + e.getMessage(), e);
        }
    }

    private boolean useRemoteRegistry() {
        String mode = readConfigValue(REGISTRY_MODE_PROP, REGISTRY_MODE_ENV, "local");
        boolean remote = REMOTE_REGISTRY_MODE.equalsIgnoreCase(mode);
        if (!remote) {
            return false;
        }
        if (!isEnterpriseMode()) {
            throw new IllegalStateException(
                    "Remote MCP registry requires enterprise mode. Enable `gollek.enterprise.enabled=true` (or `gollek.edition=enterprise`).");
        }
        return true;
    }

    private String remoteEndpoint(String suffix) {
        String configuredBase = readConfigValue(REGISTRY_API_BASE_URL_PROP, REGISTRY_API_BASE_URL_ENV, "http://localhost:8080");
        String base = configuredBase.endsWith("/") ? configuredBase.substring(0, configuredBase.length() - 1) : configuredBase;
        String path = suffix.startsWith("/") ? suffix : "/" + suffix;
        return base + REGISTRY_API_PREFIX + path;
    }

    private String readConfigValue(String prop, String env, String fallback) {
        String value = System.getProperty(prop);
        if (value == null || value.isBlank()) {
            value = System.getenv(env);
        }
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return value.trim();
    }

    private HttpRequest.Builder apiRequestBuilder(URI uri) {
        HttpRequest.Builder builder = HttpRequest.newBuilder(uri)
                .header("Accept", "application/json");
        String token = readConfigValue(REGISTRY_API_TOKEN_PROP, REGISTRY_API_TOKEN_ENV, "");
        if (token != null && !token.isBlank()) {
            builder.header("Authorization", token.startsWith("Bearer ") ? token : "Bearer " + token);
        }
        if (isEnterpriseMode() && isTenantHeaderEnabled()) {
            String tenantId = readConfigValue(TENANT_ID_PROP, TENANT_ID_ENV, "");
            if (tenantId != null && !tenantId.isBlank()) {
                builder.header(tenantHeaderName(), tenantId);
            }
        }
        return builder;
    }

    private boolean isEnterpriseMode() {
        if (parseBoolean(readConfigValue(ENTERPRISE_ENABLED_PROP, ENTERPRISE_ENABLED_ENV, "false"))) {
            return true;
        }
        if (parseBoolean(readConfigValue(MULTITENANCY_PROP, MULTITENANCY_ENV, "false"))) {
            return true;
        }
        String edition = readConfigValue(EDITION_PROP, EDITION_ENV, "community");
        return ENTERPRISE_EDITION.equalsIgnoreCase(edition);
    }

    private boolean isTenantHeaderEnabled() {
        String defaultValue = isEnterpriseMode() ? "true" : "false";
        return parseBoolean(readConfigValue(
                TENANT_HEADER_ENABLED_PROP,
                TENANT_HEADER_ENABLED_ENV,
                defaultValue));
    }

    private String tenantHeaderName() {
        String header = readConfigValue(
                TENANT_HEADER_NAME_PROP,
                TENANT_HEADER_NAME_ENV,
                "X-Tenant-Id");
        return header == null || header.isBlank() ? "X-Tenant-Id" : header;
    }

    private boolean parseBoolean(String value) {
        return value != null && ("true".equalsIgnoreCase(value) || "1".equals(value.trim()));
    }

    private String withCause(String prefix, Exception e) {
        if (e == null || e.getMessage() == null || e.getMessage().isBlank()) {
            return prefix;
        }
        return prefix + ": " + e.getMessage();
    }

    private JsonObject postJson(String url, JsonObject payload) throws IOException, InterruptedException {
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest request = apiRequestBuilder(URI.create(url))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload.toString(), StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        int status = response.statusCode();
        if (status < 200 || status >= 300) {
            throw new IllegalArgumentException("Registry API POST " + url + " returned HTTP " + status + ": " + response.body());
        }
        String body = response.body();
        if (body == null || body.isBlank()) {
            return Json.createObjectBuilder().build();
        }
        return parseJsonObject(body);
    }

    private JsonArray getJsonArray(String url) throws IOException, InterruptedException {
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest request = apiRequestBuilder(URI.create(url))
                .GET()
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        int status = response.statusCode();
        if (status < 200 || status >= 300) {
            throw new IllegalArgumentException("Registry API GET " + url + " returned HTTP " + status + ": " + response.body());
        }
        String body = response.body();
        if (body == null || body.isBlank()) {
            return Json.createArrayBuilder().build();
        }
        try (JsonReader reader = Json.createReader(new StringReader(body))) {
            return reader.readArray();
        }
    }

    private int deleteNoContent(String url) throws IOException, InterruptedException {
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest request = apiRequestBuilder(URI.create(url))
                .DELETE()
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        return response.statusCode();
    }

    private JsonObject getRemoteServer(String name) throws IOException, InterruptedException, SdkException {
        JsonArray array = getJsonArray(remoteEndpoint("/servers"));
        for (JsonValue value : array) {
            if (!(value instanceof JsonObject server)) {
                continue;
            }
            String serverName = stringValue(server.get("name"), null);
            if (name.equals(serverName)) {
                return server;
            }
        }
        throw new SdkException("SDK_ERR_MCP_NOT_FOUND", "Server not found: " + name);
    }

    private void upsertRemoteServer(String name, JsonObject server) throws IOException, InterruptedException {
        JsonObjectBuilder payload = Json.createObjectBuilder();
        String transport = resolveTransport(server);
        if (transport != null && !transport.isBlank()) {
            payload.add("transport", transport);
        }
        String command = resolveCommand(server);
        if (command != null && !command.isBlank()) {
            payload.add("command", command);
        }
        String url = resolveUrl(server);
        if (url != null && !url.isBlank()) {
            payload.add("url", url);
        }
        JsonArray args = server.getJsonArray("args");
        if (args != null) {
            payload.add("args", args);
        }
        JsonObject env = server.getJsonObject("env");
        if (env != null) {
            payload.add("env", env);
        }
        if (server.containsKey("enabled")) {
            payload.add("enabled", server.getBoolean("enabled", true));
        }
        postJson(remoteEndpoint("/servers/" + urlEncode(name)), payload.build());
    }

    private JsonArray parseJsonArrayField(JsonValue value) {
        String raw = stringValue(value, null);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        return parseJsonArray(raw);
    }

    private JsonObject parseJsonObjectField(JsonValue value) {
        String raw = stringValue(value, null);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        return parseJsonObject(raw);
    }

    private String urlEncode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    private static JsonObject parseJsonObject(String json) {
        try (JsonReader reader = Json.createReader(new StringReader(json))) {
            return reader.readObject();
        }
    }

    private static JsonArray parseJsonArray(String json) {
        try (JsonReader reader = Json.createReader(new StringReader(json))) {
            return reader.readArray();
        }
    }

    private static JsonObject extractServersObject(JsonObject root) {
        if (root == null) {
            return Json.createObjectBuilder().build();
        }
        JsonValue value = root.get(MCP_SERVERS_KEY);
        if (value instanceof JsonObject serversObj) {
            return serversObj;
        }
        return root;
    }

    private static JsonObject normalizeRoot(JsonObject root) {
        JsonObject servers = extractServersObject(root);
        return Json.createObjectBuilder().add(MCP_SERVERS_KEY, servers).build();
    }

    private static JsonObject selectServer(JsonObject servers, String serverName) {
        JsonObject selected = servers.getJsonObject(serverName);
        if (selected == null) {
            throw new IllegalArgumentException("Server not found in payload: " + serverName);
        }
        return Json.createObjectBuilder()
                .add(serverName, selected)
                .build();
    }

    private static List<String> validateServers(JsonObject servers) {
        List<String> errors = new ArrayList<>();
        servers.forEach((name, value) -> {
            if (!(value instanceof JsonObject server)) {
                errors.add(name + ": server entry must be a JSON object");
                return;
            }
            if (server.containsKey("enabled")) {
                JsonValue enabled = server.get("enabled");
                if (enabled.getValueType() != JsonValue.ValueType.TRUE
                        && enabled.getValueType() != JsonValue.ValueType.FALSE) {
                    errors.add(name + ": 'enabled' must be a boolean");
                }
            }
            String transport = resolveTransport(server);
            switch (transport) {
                case "stdio" -> {
                    if (!hasText(resolveCommand(server))) {
                        errors.add(name + ": stdio transport requires non-empty 'command'");
                    }
                    if (server.containsKey("args") && !(server.get("args") instanceof JsonArray)) {
                        errors.add(name + ": 'args' must be an array");
                    } else if (server.containsKey("args")) {
                        JsonArray arr = server.getJsonArray("args");
                        for (int i = 0; i < arr.size(); i++) {
                            if (arr.get(i).getValueType() != JsonValue.ValueType.STRING) {
                                errors.add(name + ": args[" + i + "] must be a string");
                            }
                        }
                    }
                    if (server.containsKey("env") && !(server.get("env") instanceof JsonObject)) {
                        errors.add(name + ": 'env' must be an object");
                    }
                }
                case "http", "websocket" -> {
                    if (!hasText(resolveUrl(server))) {
                        errors.add(name + ": " + transport + " transport requires non-empty 'url'");
                    }
                }
                default -> errors.add(name + ": unsupported transport '" + transport + "'");
            }
        });
        return errors;
    }

    private static JsonObject canonicalizeServers(JsonObject servers) {
        JsonObjectBuilder builder = Json.createObjectBuilder();
        servers.forEach((name, value) -> {
            if (value instanceof JsonObject server) {
                builder.add(name, canonicalizeServer(server));
            } else {
                builder.add(name, value);
            }
        });
        return builder.build();
    }

    private static JsonObject canonicalizeServer(JsonObject server) {
        JsonObjectBuilder builder = Json.createObjectBuilder();
        server.forEach(builder::add);

        String transport = resolveTransport(server);
        if (hasText(transport)) {
            builder.add("transport", transport);
        }

        String command = resolveCommand(server);
        if (hasText(command)) {
            builder.add("command", command);
        }

        String url = resolveUrl(server);
        if (hasText(url)) {
            builder.add("url", url);
        }

        return builder.build();
    }

    private static String resolveTransport(JsonObject server) {
        String explicit = firstNonBlankString(server, "transport", "type");
        if (hasText(explicit)) {
            return explicit.toLowerCase();
        }
        String url = resolveUrl(server);
        if (hasText(url)) {
            String lower = url.toLowerCase();
            if (lower.startsWith("ws://") || lower.startsWith("wss://")) {
                return "websocket";
            }
            return "http";
        }
        return "stdio";
    }

    private static String resolveCommand(JsonObject server) {
        return firstNonBlankString(server, "command", "cmd", "executable");
    }

    private static String resolveUrl(JsonObject server) {
        return firstNonBlankString(server, "url", "serverUrl", "endpoint");
    }

    private static String firstNonBlankString(JsonObject object, String... keys) {
        for (String key : keys) {
            String value = stringValue(object.get(key), null);
            if (hasText(value)) {
                return value.trim();
            }
        }
        return null;
    }

    private static String stringValue(JsonValue value, String fallback) {
        if (value == null) {
            return fallback;
        }
        if (value.getValueType() == JsonValue.ValueType.STRING) {
            return ((JsonString) value).getString();
        }
        if (value.getValueType() == JsonValue.ValueType.NULL) {
            return fallback;
        }
        return value.toString();
    }

    private static boolean hasNonBlankString(JsonValue value) {
        return value != null
                && value.getValueType() == JsonValue.ValueType.STRING
                && !((JsonString) value).getString().isBlank();
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    private String requiredString(JsonValue value, String message) {
        String str = stringValue(value, null);
        if (str == null || str.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return str;
    }

    private List<String> readStringArrayAsList(JsonValue value) {
        if (!(value instanceof JsonArray arr)) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        arr.forEach(item -> {
            if (item.getValueType() == JsonValue.ValueType.STRING) {
                out.add(((JsonString) item).getString());
            }
        });
        return out;
    }

    private String jsonScalarToString(JsonValue value) {
        return switch (value.getValueType()) {
            case STRING -> ((JsonString) value).getString();
            case TRUE -> "true";
            case FALSE -> "false";
            case NUMBER -> value.toString();
            default -> value.toString();
        };
    }

    private void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
