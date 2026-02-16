package tech.kayys.wayang.tool.entity;

import java.time.Instant;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import jakarta.persistence.CollectionTable;
import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import jakarta.validation.constraints.NotNull;
import tech.kayys.wayang.tool.dto.CapabilityLevel;
import tech.kayys.wayang.tool.dto.ToolType;

/**
 * MCP Tool - Represents an executable tool derived from OpenAPI
 */
@Entity
@Table(name = "mcp_tools", indexes = {
        @Index(name = "idx_mcp_tools_tenant", columnList = "tenant_id"),
        @Index(name = "idx_mcp_tools_namespace", columnList = "namespace"),
        @Index(name = "idx_mcp_tools_enabled", columnList = "enabled"),
        @Index(name = "idx_mcp_tools_composite", columnList = "tenant_id,namespace,name")
})
public class McpTool extends PanacheEntityBase {

    @Id
    @Column(name = "tool_id")
    private String toolId; // Format: {namespace}.{operationId}

    @NotNull
    @Column(name = "tenant_id")
    private String requestId;

    @NotNull
    @Column(name = "namespace")
    private String namespace; // e.g., "payment-api", "crm-api"

    @NotNull
    @Column(name = "name")
    private String name; // e.g., "createPayment"

    @Column(name = "version")
    private String version = "1.0.0";

    @Column(name = "description", length = 1000)
    private String description;

    @Enumerated(EnumType.STRING)
    @Column(name = "tool_type")
    private ToolType toolType = ToolType.HTTP;

    @Enumerated(EnumType.STRING)
    @Column(name = "capability_level")
    private CapabilityLevel capabilityLevel;

    // JSON Schema for input validation
    @Column(name = "input_schema", columnDefinition = "jsonb")
    private Map<String, Object> inputSchema;

    // JSON Schema for output validation
    @Column(name = "output_schema", columnDefinition = "jsonb")
    private Map<String, Object> outputSchema;

    // HTTP execution configuration
    @Embedded
    private HttpExecutionConfig executionConfig;

    // Authentication reference
    @Column(name = "auth_profile_id")
    private String authProfileId;

    // Safety & guardrails
    @Embedded
    private ToolGuardrails guardrails;

    // Operational metadata
    @Column(name = "enabled")
    private boolean enabled = true;

    @Column(name = "read_only")
    private boolean readOnly = false;

    @Column(name = "requires_approval")
    private boolean requiresApproval = false;

    // Categorization & discovery
    @ElementCollection
    @CollectionTable(name = "mcp_tool_tags")
    private Set<String> tags = new HashSet<>();

    @ElementCollection
    @CollectionTable(name = "mcp_tool_capabilities")
    private Set<String> capabilities = new HashSet<>();

    // Temporal tracking
    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;

    @Column(name = "created_by")
    private String createdBy;

    // OpenAPI source reference
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "openapi_source_id")
    private OpenApiSource source;

    @Column(name = "operation_id")
    private String operationId; // Original OpenAPI operationId

    // Metrics & usage
    @Embedded
    private ToolMetrics metrics;

    // Versioning
    @Version
    @Column(name = "version_number")
    private Long versionNumber;

    // Getters and Setters
    public String getToolId() {
        return toolId;
    }

    public void setToolId(String toolId) {
        this.toolId = toolId;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getNamespace() {
        return namespace;
    }

    public void setNamespace(String namespace) {
        this.namespace = namespace;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public ToolType getToolType() {
        return toolType;
    }

    public void setToolType(ToolType toolType) {
        this.toolType = toolType;
    }

    public CapabilityLevel getCapabilityLevel() {
        return capabilityLevel;
    }

    public void setCapabilityLevel(CapabilityLevel capabilityLevel) {
        this.capabilityLevel = capabilityLevel;
    }

    public Map<String, Object> getInputSchema() {
        return inputSchema;
    }

    public void setInputSchema(Map<String, Object> inputSchema) {
        this.inputSchema = inputSchema;
    }

    public Map<String, Object> getOutputSchema() {
        return outputSchema;
    }

    public void setOutputSchema(Map<String, Object> outputSchema) {
        this.outputSchema = outputSchema;
    }

    public HttpExecutionConfig getExecutionConfig() {
        return executionConfig;
    }

    public void setExecutionConfig(HttpExecutionConfig executionConfig) {
        this.executionConfig = executionConfig;
    }

    public String getAuthProfileId() {
        return authProfileId;
    }

    public void setAuthProfileId(String authProfileId) {
        this.authProfileId = authProfileId;
    }

    public ToolGuardrails getGuardrails() {
        return guardrails;
    }

    public void setGuardrails(ToolGuardrails guardrails) {
        this.guardrails = guardrails;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isReadOnly() {
        return readOnly;
    }

    public void setReadOnly(boolean readOnly) {
        this.readOnly = readOnly;
    }

    public boolean isRequiresApproval() {
        return requiresApproval;
    }

    public void setRequiresApproval(boolean requiresApproval) {
        this.requiresApproval = requiresApproval;
    }

    public Set<String> getTags() {
        return tags;
    }

    public void setTags(Set<String> tags) {
        this.tags = tags;
    }

    public Set<String> getCapabilities() {
        return capabilities;
    }

    public void setCapabilities(Set<String> capabilities) {
        this.capabilities = capabilities;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    public String getCreatedBy() {
        return createdBy;
    }

    public void setCreatedBy(String createdBy) {
        this.createdBy = createdBy;
    }

    public OpenApiSource getSource() {
        return source;
    }

    public void setSource(OpenApiSource source) {
        this.source = source;
    }

    public String getOperationId() {
        return operationId;
    }

    public void setOperationId(String operationId) {
        this.operationId = operationId;
    }

    public ToolMetrics getMetrics() {
        return metrics;
    }

    public void setMetrics(ToolMetrics metrics) {
        this.metrics = metrics;
    }

    public Long getVersionNumber() {
        return versionNumber;
    }

    public void setVersionNumber(Long versionNumber) {
        this.versionNumber = versionNumber;
    }
}
