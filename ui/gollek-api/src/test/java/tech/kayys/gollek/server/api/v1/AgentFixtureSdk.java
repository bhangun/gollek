package tech.kayys.gollek.server.api.v1;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.mcp.McpAddRequest;
import tech.kayys.gollek.sdk.mcp.McpDoctorReport;
import tech.kayys.gollek.sdk.mcp.McpEditRequest;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.mcp.McpServerSummary;
import tech.kayys.gollek.sdk.mcp.McpServerView;
import tech.kayys.gollek.sdk.mcp.McpTestReport;
import tech.kayys.gollek.sdk.mcp.McpToolModel;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

final class AgentFixtureSdk {
    private AgentFixtureSdk() {
    }

    static GollekSdk serving() {
        return (GollekSdk) Proxy.newProxyInstance(
                GollekSdk.class.getClassLoader(),
                new Class<?>[] { GollekSdk.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "getModelInfo" -> Optional.of(modelInfo((String) args[0]));
                    case "listAvailableProviders" -> List.of();
                    case "getPreferredProvider" -> Optional.empty();
                    case "mcpRegistry" -> new FixtureMcpRegistry();
                    case "toString" -> "AgentFixtureSdk";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }

    private static ModelInfo modelInfo(String modelId) {
        String effectiveModelId = modelId == null || modelId.isBlank() ? "fixture-model" : modelId;
        boolean embedding = looksLikeEmbeddingModel(effectiveModelId);
        ModelInfo.Builder builder = ModelInfo.builder()
                .modelId(effectiveModelId)
                .name("Fixture " + effectiveModelId)
                .architecture(embedding ? "fixture-embedding" : "fixture-transformer")
                .format("SAFETENSORS")
                .contextLength(8192L)
                .outputTokenLimit(1024)
                .metadata(embedding
                        ? Map.of("embeddings", true)
                        : Map.of("tool_calling", true));
        if (embedding) {
            builder.embeddingSize(768);
        }
        return builder.build();
    }

    private static boolean looksLikeEmbeddingModel(String modelId) {
        String normalized = modelId.toLowerCase(Locale.ROOT);
        return normalized.contains("embed") || normalized.contains("embedding");
    }

    private static final class FixtureMcpRegistry implements McpRegistryManager {
        @Override
        public String registryPath() {
            return "/fixture/gollek/mcp.json";
        }

        @Override
        public List<McpServerSummary> list() {
            return List.of(new McpServerSummary("knowledge", true));
        }

        @Override
        public McpServerView show(String name) throws SdkException {
            return new McpServerView(name, true, "stdio", "npx", 1, 0, null, "{}");
        }

        @Override
        public List<McpToolModel> listTools(String name) {
            return List.of(new McpToolModel(
                    "search",
                    "Search indexed knowledge",
                    Map.of(
                            "type", "object",
                            "required", List.of("query"),
                            "properties", Map.of(
                                    "query", Map.of("type", "string")))));
        }

        @Override
        public List<String> add(McpAddRequest request) {
            throw unsupportedMutation();
        }

        @Override
        public void remove(String name) {
            throw unsupportedMutation();
        }

        @Override
        public void rename(String oldName, String newName) {
            throw unsupportedMutation();
        }

        @Override
        public void edit(McpEditRequest request) {
            throw unsupportedMutation();
        }

        @Override
        public void setEnabled(String name, boolean enabled) {
            throw unsupportedMutation();
        }

        @Override
        public int importFromFile(String filePath, boolean replace) {
            throw unsupportedMutation();
        }

        @Override
        public int exportToFile(String filePath, String name) {
            throw unsupportedMutation();
        }

        @Override
        public McpDoctorReport doctor() {
            throw unsupportedMutation();
        }

        @Override
        public McpTestReport test(String name, boolean all, long timeoutMs) {
            throw unsupportedMutation();
        }

        private UnsupportedOperationException unsupportedMutation() {
            return new UnsupportedOperationException("fixture registry is read-only");
        }
    }
}
