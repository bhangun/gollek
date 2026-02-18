package tech.kayys.gollek.cli.commands;

import jakarta.json.Json;
import jakarta.json.JsonObject;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class McpCommandTest {

    @TempDir
    Path tempHome;

    @Test
    void addCreatesRegistryAndStoresServer() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            add.run();

            Path registry = Path.of(System.getProperty("user.home"), ".gollek", "mcp", "servers.json");
            assertTrue(Files.exists(registry));

            JsonObject root = McpCommand.loadRegistry();
            assertTrue(root.getJsonObject("mcpServers").containsKey("image-downloader"));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void removeDeletesNamedServerFromRegistry() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]},\"other\":{\"command\":\"node\",\"args\":[\"/tmp/other.js\"]}}}";
            add.run();

            McpCommand.Remove remove = new McpCommand.Remove();
            remove.name = "image-downloader";
            remove.run();

            JsonObject root = McpCommand.loadRegistry();
            JsonObject servers = root.getJsonObject("mcpServers");
            assertFalse(servers.containsKey("image-downloader"));
            assertTrue(servers.containsKey("other"));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void validateServersRejectsInvalidStdioEntry() {
        JsonObject servers = Json.createObjectBuilder()
                .add("bad", Json.createObjectBuilder()
                        .add("transport", "stdio")
                        .build())
                .build();

        List<String> errors = McpCommand.validateServers(servers);
        assertEquals(1, errors.size());
        assertTrue(errors.get(0).contains("requires non-empty 'command'"));
    }

    @Test
    void exportWritesRegistryJsonFile() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            add.run();

            Path out = tempHome.resolve("exported-mcp.json");
            McpCommand.ExportServers export = new McpCommand.ExportServers();
            export.filePath = out.toString();
            export.run();

            assertTrue(Files.exists(out));
            JsonObject exported = McpCommand.parseJsonObject(Files.readString(out));
            assertTrue(exported.getJsonObject("mcpServers").containsKey("image-downloader"));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void importReplaceReplacesExistingRegistry() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"old\":{\"command\":\"node\",\"args\":[\"/tmp/old.js\"]}}}";
            add.run();

            Path importFile = tempHome.resolve("import.json");
            Files.writeString(importFile, """
                    {
                      "mcpServers": {
                        "new": {
                          "command": "node",
                          "args": ["/tmp/new.js"]
                        }
                      }
                    }
                    """);

            McpCommand.ImportServers importServers = new McpCommand.ImportServers();
            importServers.filePath = importFile.toString();
            importServers.replace = true;
            importServers.merge = false;
            importServers.run();

            JsonObject servers = McpCommand.loadRegistry().getJsonObject("mcpServers");
            assertFalse(servers.containsKey("old"));
            assertTrue(servers.containsKey("new"));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void renameChangesServerKey() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"old-name\":{\"command\":\"node\",\"args\":[\"/tmp/a.js\"]}}}";
            add.run();

            McpCommand.RenameServer rename = new McpCommand.RenameServer();
            rename.oldName = "old-name";
            rename.newName = "new-name";
            rename.run();

            JsonObject servers = McpCommand.loadRegistry().getJsonObject("mcpServers");
            assertFalse(servers.containsKey("old-name"));
            assertTrue(servers.containsKey("new-name"));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void disableAndEnableToggleServerFlag() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            add.run();

            McpCommand.DisableServer disable = new McpCommand.DisableServer();
            disable.name = "image-downloader";
            disable.run();

            JsonObject server = McpCommand.loadRegistry().getJsonObject("mcpServers").getJsonObject("image-downloader");
            assertFalse(server.getBoolean("enabled", true));

            McpCommand.EnableServer enable = new McpCommand.EnableServer();
            enable.name = "image-downloader";
            enable.run();

            server = McpCommand.loadRegistry().getJsonObject("mcpServers").getJsonObject("image-downloader");
            assertTrue(server.getBoolean("enabled", false));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void testRequiresNameOrAll() {
        PrintStream originalErr = System.err;
        ByteArrayOutputStream errBuffer = new ByteArrayOutputStream();
        System.setErr(new PrintStream(errBuffer));
        try {
            McpCommand.TestServer testServer = new McpCommand.TestServer();
            testServer.timeoutMs = 50;
            testServer.run();
            assertTrue(errBuffer.toString().contains("Provide <name> or use --all."));
        } finally {
            System.setErr(originalErr);
        }
    }

    @Test
    void editUpdatesServerFields() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            add.run();

            McpCommand.EditServer edit = new McpCommand.EditServer();
            edit.name = "image-downloader";
            edit.command = "node2";
            edit.argsJson = "[\"/tmp/new-index.js\",\"--verbose\"]";
            edit.enabled = false;
            edit.run();

            JsonObject server = McpCommand.loadRegistry()
                    .getJsonObject("mcpServers")
                    .getJsonObject("image-downloader");
            assertEquals("node2", server.getString("command"));
            assertEquals(2, server.getJsonArray("args").size());
            assertFalse(server.getBoolean("enabled", true));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void editRejectsArgsConflict() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        PrintStream originalErr = System.err;
        ByteArrayOutputStream errBuffer = new ByteArrayOutputStream();
        System.setErr(new PrintStream(errBuffer));
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            add.run();

            McpCommand.EditServer edit = new McpCommand.EditServer();
            edit.name = "image-downloader";
            edit.argsJson = "[\"/tmp/new-index.js\"]";
            edit.clearArgs = true;
            edit.run();

            assertTrue(errBuffer.toString().contains("Use either --args-json or --clear-args, not both."));
        } finally {
            System.setErr(originalErr);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void doctorReportsValidAndInvalidServers() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outBuffer = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outBuffer));
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = """
                    {
                      "mcpServers": {
                        "ok": {
                          "command": "node",
                          "args": ["/tmp/index.js"]
                        }
                      }
                    }
                    """;
            add.run();

            Path registry = Path.of(System.getProperty("user.home"), ".gollek", "mcp", "servers.json");
            Files.writeString(registry, """
                    {
                      "mcpServers": {
                        "ok": {
                          "command": "node",
                          "args": ["/tmp/index.js"]
                        },
                        "bad": {
                          "transport": "stdio"
                        }
                      }
                    }
                    """);

            McpCommand.DoctorServers doctor = new McpCommand.DoctorServers();
            doctor.run();

            String out = outBuffer.toString();
            assertTrue(out.contains("[OK] ok"));
            assertTrue(out.contains("[ERROR] bad"));
            assertTrue(out.contains("MCP doctor summary: passed=1 failed=1 total=2"));
        } finally {
            System.setOut(originalOut);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void doctorHandlesEmptyRegistry() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outBuffer = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outBuffer));
        try {
            McpCommand.DoctorServers doctor = new McpCommand.DoctorServers();
            doctor.run();
            assertTrue(outBuffer.toString().contains("No MCP servers configured."));
        } finally {
            System.setOut(originalOut);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void addSupportsStructuredFlags() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.name = "image-downloader";
            add.command = "node";
            add.argsJson = "[\"/tmp/index.js\"]";
            add.enabled = true;
            add.run();

            JsonObject server = McpCommand.loadRegistry()
                    .getJsonObject("mcpServers")
                    .getJsonObject("image-downloader");
            assertEquals("node", server.getString("command"));
            assertEquals(1, server.getJsonArray("args").size());
            assertTrue(server.getBoolean("enabled", false));
        } finally {
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void addRejectsMixingJsonAndStructuredFlags() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        PrintStream originalErr = System.err;
        ByteArrayOutputStream errBuffer = new ByteArrayOutputStream();
        System.setErr(new PrintStream(errBuffer));
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.inlineJson = "{\"mcpServers\":{\"x\":{\"command\":\"node\"}}}";
            add.name = "y";
            add.command = "node";
            add.run();

            assertTrue(errBuffer.toString().contains("Do not combine JSON input"));
        } finally {
            System.setErr(originalErr);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void addSupportsFromUrl() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/mcp.json", exchange -> {
            String body = "{\"mcpServers\":{\"image-downloader\":{\"command\":\"node\",\"args\":[\"/tmp/index.js\"]}}}";
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/mcp.json";
            McpCommand.Add add = new McpCommand.Add();
            add.fromUrl = url;
            add.run();

            JsonObject serverObj = McpCommand.loadRegistry()
                    .getJsonObject("mcpServers")
                    .getJsonObject("image-downloader");
            assertEquals("node", serverObj.getString("command"));
            assertEquals(1, serverObj.getJsonArray("args").size());
        } finally {
            server.stop(0);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }

    @Test
    void addRejectsMixingFromUrlWithOtherInputs() throws Exception {
        String oldHome = System.getProperty("user.home");
        System.setProperty("user.home", tempHome.toString());
        PrintStream originalErr = System.err;
        ByteArrayOutputStream errBuffer = new ByteArrayOutputStream();
        System.setErr(new PrintStream(errBuffer));
        try {
            McpCommand.Add add = new McpCommand.Add();
            add.fromUrl = "http://127.0.0.1:9/mcp.json";
            add.name = "image-downloader";
            add.command = "node";
            add.run();
            assertTrue(errBuffer.toString().contains("Do not combine --from-url with other add inputs."));
        } finally {
            System.setErr(originalErr);
            if (oldHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", oldHome);
            }
        }
    }
}
