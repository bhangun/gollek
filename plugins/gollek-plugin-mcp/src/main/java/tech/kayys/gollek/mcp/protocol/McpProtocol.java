package tech.kayys.gollek.mcp.protocol;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Protocol types for the Model Context Protocol (MCP) 2025-11-25 specification.
 *
 * <p>MCP is a standard for connecting AI assistants to external data sources
 * and tools. It uses JSON-RPC 2.0 over SSE (Server-Sent Events), stdio, or HTTP Streamable.
 * See: <a href="https://modelcontextprotocol.io/specification/2025-11-25">MCP Spec 2025-11-25</a></p>
 *
 * <h2>Message flow</h2>
 * <pre>
 * Client → Server: initialize request
 * Server → Client: initialize response (capabilities)
 * Client → Server: initialized notification
 * Client → Server: tools/list request
 * Server → Client: tools/list response
 * Client → Server: tools/call request
 * Server → Client: tools/call response
 * </pre>
 *
 * <h2>Compliance notes</h2>
 * <ul>
 *   <li>JSON Schema dialect defaults to 2020-12</li>
 *   <li>_meta parameter reserved for protocol-level metadata</li>
 *   <li>All list endpoints support pagination via cursors</li>
 *   <li>Capabilities negotiation in initialize handshake</li>
 *   <li>Support for roots, sampling, logging, completions, elicitation</li>
 * </ul>
 */
public final class McpProtocol {

    private McpProtocol() {}

    public static final String JSONRPC_VERSION = "2.0";
    public static final String MCP_VERSION     = "2025-11-25";

    // ── JSON-RPC base types ────────────────────────────────────────────────────

    /**
     * JSON-RPC 2.0 request.
     * Per spec: id MUST be non-null and unique per session.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record JsonRpcRequest(
            String jsonrpc,
            String id,
            String method,
            Object params
    ) {
        public JsonRpcRequest(String id, String method, Object params) {
            this(JSONRPC_VERSION, id, method, params);
        }
    }

    /**
     * JSON-RPC 2.0 notification (no response expected).
     * MUST NOT include an id field.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record JsonRpcNotification(
            String jsonrpc,
            String method,
            Object params
    ) {
        public JsonRpcNotification(String method, Object params) {
            this(JSONRPC_VERSION, method, params);
        }
    }

    /**
     * JSON-RPC 2.0 successful response.
     * id MUST match the corresponding request.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record JsonRpcResponse(
            String jsonrpc,
            String id,
            JsonNode result,
            JsonRpcError error
    ) {}

    /**
     * JSON-RPC 2.0 error response.
     * code MUST be an integer. Per spec: -32700 Parse error, -32600 Invalid Request,
     * -32601 Method not found, -32602 Invalid params, -32603 Internal error,
     * -32000 to -32099 Server-reserved (custom extensions).
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record JsonRpcError(int code, String message, Object data) {}

    // ── Protocol metadata (_meta) ─────────────────────────────────────────────

    /**
     * Reserved parameter key for protocol-level metadata.
     * Key format: prefix/name where prefix is reverse DNS (e.g., com.example/).
     * Reserved for MCP if second label is "modelcontextprotocol" or "mcp".
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record MetaPayload(Map<String, Object> metadata) {
        public static final String MCP_PREFIX = "io.modelcontextprotocol/";
    }

    // ── Initialize ────────────────────────────────────────────────────────────

    /**
     * Initialize handshake params from client.
     * Client MUST send protocolVersion, capabilities, and clientInfo.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record InitializeParams(
            String protocolVersion,
            ClientCapabilities capabilities,
            ClientInfo clientInfo
    ) {
        public static InitializeParams of(String clientName, String clientVersion) {
            return new InitializeParams(
                    MCP_VERSION,
                    new ClientCapabilities(
                            new RootsCapability(false),
                            new SamplingCapability(),
                            new ElicitationCapability()
                    ),
                    new ClientInfo(clientName, clientVersion)
            );
        }

        public static InitializeParams of(String clientName, String clientVersion,
                                          boolean rootsListChanged, boolean samplingEnabled) {
            return new InitializeParams(
                    MCP_VERSION,
                    new ClientCapabilities(
                            rootsListChanged ? new RootsCapability(true) : null,
                            samplingEnabled ? new SamplingCapability() : null,
                            new ElicitationCapability()
                    ),
                    new ClientInfo(clientName, clientVersion)
            );
        }
    }

    public record ClientInfo(String name, String version) {}

    /**
     * Client capabilities advertised during initialization.
     * All fields are optional - only include supported capabilities.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ClientCapabilities(
            RootsCapability roots,
            SamplingCapability sampling,
            ElicitationCapability elicitation
    ) {}

    /**
     * Roots capability - client will expose file system roots to server.
     * listChanged: true if client emits notifications when root list changes.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record RootsCapability(boolean listChanged) {}

    /**
     * Sampling capability - client allows server to request LLM sampling.
     * Empty object indicates support.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record SamplingCapability() {}

    /**
     * Elicitation capability - client allows server to request user input.
     * Empty object indicating support.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ElicitationCapability() {}

    /**
     * Initialize response from server.
     * MUST include protocolVersion, capabilities, and serverInfo.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record InitializeResult(
            String protocolVersion,
            ServerCapabilities capabilities,
            ServerInfo serverInfo,
            String instructions
    ) {}

    /**
     * Server capabilities negotiated during initialization.
     * May include: tools, resources, prompts, logging, completions.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ServerCapabilities(
            ToolsCapability tools,
            ResourcesCapability resources,
            PromptsCapability prompts,
            LoggingCapability logging,
            CompletionsCapability completions
    ) {
        public boolean hasTools()     { return tools     != null; }
        public boolean hasResources() { return resources != null; }
        public boolean hasPrompts()   { return prompts   != null; }
        public boolean hasLogging()   { return logging   != null; }
        public boolean hasCompletions() { return completions != null; }
    }

    /** Tools capability - listChanged: true if server sends notifications/tools/list_changed */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ToolsCapability(boolean listChanged) {}

    /**
     * Resources capability.
     * subscribe: true if server supports resources/subscribe and /unsubscribe.
     * listChanged: true if server sends notifications/resources/list_changed.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ResourcesCapability(boolean subscribe, boolean listChanged) {}

    /** Prompts capability - listChanged: true if server sends notifications/prompts/list_changed */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptsCapability(boolean listChanged) {}

    /** Logging capability - empty object indicates support */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record LoggingCapability() {}

    /** Completions capability - empty object indicates support for completion/complete */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompletionsCapability() {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ServerInfo(String name, String version) {}

    // ── Tools ─────────────────────────────────────────────────────────────────

    /**
     * Tools list request params.
     * cursor: optional pagination cursor from previous response.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ListToolsParams(String cursor) {
        public ListToolsParams() { this(null); }
    }

    /**
     * Tools list response.
     * nextCursor: omitted when last page reached.
     * Empty tools array with no nextCursor indicates completion.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ListToolsResult(List<McpTool> tools, String nextCursor) {}

    /**
     * Tool definition.
     * inputSchema: JSON Schema 2020-12 object for tool arguments.
     * annotations: optional hints about tool behavior.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record McpTool(
            String name,
            String description,
            InputSchema inputSchema,
            ToolAnnotations annotations
    ) {}

    /**
     * Tool annotations - hints for clients about tool behavior.
     * Per spec: MUST be treated as untrusted unless from verified server.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ToolAnnotations(
            String title,
            Boolean readOnlyHint,
            Boolean destructiveHint,
            Boolean idempotentHint,
            Boolean openWorldHint
    ) {}

    /**
     * Tool input schema.
     * type: MUST be "object".
     * properties: JSON Schema 2020-12 property definitions.
     * required: array of required property names.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record InputSchema(
            String type,
            Map<String, Object> properties,
            List<String> required
    ) {
        public InputSchema(Map<String, Object> properties, List<String> required) {
            this("object", properties, required);
        }
    }

    /**
     * Tool call request params.
     * arguments: optional map matching tool's inputSchema.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CallToolParams(String name, Map<String, Object> arguments) {}

    /**
     * Tool call response.
     * content: array of content blocks (text, image, resource).
     * isError: true if tool execution failed (client should handle gracefully).
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CallToolResult(List<ContentBlock> content, Boolean isError) {
        /** Extract the first text content block's text value. */
        public String firstText() {
            if (content == null || content.isEmpty()) return "";
            return content.stream()
                    .filter(c -> "text".equals(c.type()))
                    .map(ContentBlock::text)
                    .findFirst()
                    .orElse("");
        }
        /** Combines all text content blocks. */
        public String allText() {
            if (content == null) return "";
            return content.stream()
                    .filter(c -> "text".equals(c.type()) && c.text() != null)
                    .map(ContentBlock::text)
                    .reduce("", (a, b) -> a.isEmpty() ? b : a + "\n" + b);
        }
        public boolean failed() { return Boolean.TRUE.equals(isError); }
    }

    /**
     * Content block - can be text, image, audio, or embedded resource.
     * type: "text" | "image" | "audio" | "resource"
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ContentBlock(
            String type,       // "text" | "image" | "audio" | "resource"
            String text,       // for type="text"
            String mimeType,   // for type="image", "audio", or "resource"
            String data,       // base64 for type="image" or "audio"
            ResourceContent resource,  // for type="resource"
            String id          // for type="audio" - optional identifier
    ) {}

    // ── Resources ─────────────────────────────────────────────────────────────

    /** Resources list request params with optional cursor */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ListResourcesParams(String cursor) {
        public ListResourcesParams() { this(null); }
    }

    /** Resources list response */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ListResourcesResult(List<McpResource> resources, String nextCursor) {}

    /**
     * Resource definition.
     * uri: unique identifier for the resource.
     * mimeType: optional MIME type hint.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record McpResource(String uri, String name, String description, String mimeType, List<ResourceIcon> icons) {
        public McpResource(String uri, String name, String description, String mimeType) {
            this(uri, name, description, mimeType, null);
        }
    }

    /**
     * Resource icon.
     * src: MUST use https: or data: schemes.
     * Clients MUST support image/png and image/jpeg.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ResourceIcon(
            String src,
            String mimeType,
            List<String> sizes,
            String theme  // "light" | "dark"
    ) {}

    /** Read resource request params */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ReadResourceParams(String uri, String _meta) {
        public ReadResourceParams(String uri) {
            this(uri, null);
        }
    }

    /**
     * Read resource response.
     * contents: array of resource contents (text or blob).
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ReadResourceResult(List<ResourceContent> contents) {}

    /**
     * Embedded resource content (within content blocks).
     * uri: resource URI.
     * text or blob: exactly one must be present.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ResourceContent(
            String uri,
            String mimeType,
            String text,
            String blob  // base64-encoded binary content
    ) {}

    // ── Resource subscriptions ────────────────────────────────────────────────

    /** Subscribe to resource updates */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record SubscribeParams(String uri) {}

    /** Unsubscribe from resource updates */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record UnsubscribeParams(String uri) {}

    // ── Prompts ───────────────────────────────────────────────────────────────

    /** Prompts list request params with optional cursor */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ListPromptsParams(String cursor) {
        public ListPromptsParams() { this(null); }
    }

    /** Prompts list response */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ListPromptsResult(List<McpPrompt> prompts, String nextCursor) {}

    /**
     * Prompt definition.
     * arguments: optional list of arguments the prompt accepts.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record McpPrompt(String name, String description, List<PromptArgument> arguments) {}

    /**
     * Prompt argument definition.
     * required: whether this argument must be provided.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptArgument(String name, String description, boolean required) {}

    /**
     * Get prompt request params.
     * arguments: map of argument values matching prompt's argument definitions.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record GetPromptParams(String name, Map<String, String> arguments) {}

    /**
     * Get prompt response.
     * messages: array of messages with role and content.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record GetPromptResult(String description, List<PromptMessage> messages) {}

    /**
     * Prompt message.
     * role: "user" | "assistant".
     * content: text or image content.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptMessage(String role, ContentBlock content) {}

    // ── Roots ─────────────────────────────────────────────────────────────────

    /**
     * Roots list request (server → client).
     * No params required.
     */
    public record RootsListRequest() {}

    /**
     * Roots list response (client → server).
     * roots: array of exposed file system roots.
     * URI MUST be file:// scheme.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RootsResult(List<RootInfo> roots) {}

    /**
     * Root info.
     * uri: MUST be file:// URI.
     * name: optional human-readable name.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RootInfo(String uri, String name) {}

    // ── Sampling ──────────────────────────────────────────────────────────────

    /**
     * Sampling request (server → client).
     * Server requests client to perform LLM sampling.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CreateMessageParams(
            List<SamplingMessage> messages,
            String systemPrompt,
            IncludeContext includeContext,
            String temperature,
            Integer maxTokens,
            List<String> stopSequences,
            Map<String, Object> metadata
    ) {
        public enum IncludeContext {
            none, allServers, thisServer
        }
    }

    /**
     * Message for sampling request.
     * role: "user" | "assistant".
     * content: text or image content.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record SamplingMessage(String role, ContentBlock content) {}

    /**
     * Sampling response (client → server).
     * Contains the LLM's response.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CreateMessageResult(
            String role,
            ContentBlock content,
            String model,
            String stopReason  // "endTurn" | "stopSequence" | "maxTokens" | "cancelled" | null
    ) {}

    // ── Elicitation ───────────────────────────────────────────────────────────

    /**
     * Elicitation request (server → client).
     * Server requests user input from client.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ElicitRequest(
            String message,
            String requestedSchema
    ) {}

    /**
     * Elicitation response (client → server).
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ElicitResult(String action, Map<String, Object> content) {}

    // ── Completions ───────────────────────────────────────────────────────────

    /**
     * Completion request.
     * Provides argument auto-completion for prompts/resources.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CompleteParams(CompleteRef ref, CompleteArgument argument) {}

    /** Reference to prompt or resource for completion */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompleteRef(String type, String name, String uri) {
        public static CompleteRef forPrompt(String name) {
            return new CompleteRef("ref/prompt", name, null);
        }
        public static CompleteRef forResource(String uri) {
            return new CompleteRef("ref/resource", null, uri);
        }
    }

    /** Argument value to complete */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompleteArgument(String name, String value) {}

    /**
     * Completion response.
     * values: array of completion values.
     * hasMore: true if more values available.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompleteResult(CompleteInfo completion) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompleteInfo(List<String> values, Integer total, Boolean hasMore) {}

    // ── Logging ───────────────────────────────────────────────────────────────

    /**
     * Set logging level (client → server).
     * level: minimum severity to receive.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record SetLoggingLevelParams(String level) {}

    /**
     * Log message notification (server → client).
     * method: notifications/message
     * level: debug | info | notice | warning | error | critical | alert | emergency
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record LoggingMessageNotification(
            String level,
            String logger,
            Object data
    ) {
        public static final String METHOD = "notifications/message";
    }

    // ── Progress tracking ─────────────────────────────────────────────────────

    /**
     * Progress notification.
     * Sent during long-running operations.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ProgressNotification(
            Object progressToken,  // from _meta.progressToken
            Integer progress,
            Integer total
    ) {
        public static final String METHOD = "notifications/progress";
    }

    // ── Cancellation ──────────────────────────────────────────────────────────

    /**
     * Cancel notification.
     * requestId: ID of request to cancel.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CancelledNotification(
            String requestId,
            String reason
    ) {
        public static final String METHOD = "notifications/cancelled";
    }

    // ── List changed notifications ────────────────────────────────────────────

    /** Notification: tools list changed */
    public record ToolsListChangedNotification() {
        public static final String METHOD = "notifications/tools/list_changed";
    }

    /** Notification: resources list changed */
    public record ResourcesListChangedNotification() {
        public static final String METHOD = "notifications/resources/list_changed";
    }

    /** Notification: prompts list changed */
    public record PromptsListChangedNotification() {
        public static final String METHOD = "notifications/prompts/list_changed";
    }

    /** Notification: resource updated */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ResourceUpdatedNotification(String uri) {
        public static final String METHOD = "notifications/resources/updated";
    }

    /** Notification: roots list changed */
    public record RootsListChangedNotification() {
        public static final String METHOD = "notifications/roots/list_changed";
    }

    // ── Notifications ─────────────────────────────────────────────────────────

    /** Client sends this after receiving initialize response */
    public record InitializedNotification() {}

    // ── Error codes (JSON-RPC + MCP-specific) ────────────────────────────────

    /** Standard JSON-RPC 2.0 error codes */
    public static final int ERR_PARSE_ERROR       = -32700;
    public static final int ERR_INVALID_REQUEST   = -32600;
    public static final int ERR_METHOD_NOT_FOUND  = -32601;
    public static final int ERR_INVALID_PARAMS    = -32602;
    public static final int ERR_INTERNAL          = -32603;

    /** MCP-specific error codes (-32000 to -32099 server-reserved) */
    public static final int ERR_TIMEOUT           = -32000; // Request timeout
    public static final int ERR_NOT_FOUND         = -32002; // MCP: resource/tool/prompt not found
    public static final int ERR_INVALID_CURSOR    = -32001; // MCP: invalid pagination cursor
}
