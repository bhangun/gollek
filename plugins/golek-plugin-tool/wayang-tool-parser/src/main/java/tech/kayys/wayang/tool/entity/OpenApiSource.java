package tech.kayys.wayang.tool.entity;

import io.quarkus.hibernate.reactive.panache.PanacheEntityBase;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotNull;
import tech.kayys.wayang.tool.dto.SourceStatus;
import tech.kayys.wayang.tool.dto.SourceType;

/**
 * OpenAPI source document
 * Tracks the original OpenAPI spec from which tools were generated
 */
@Entity
@Table(name = "mcp_openapi_sources")
public class OpenApiSource extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(name = "source_id")
    private UUID sourceId;

    @NotNull
    @Column(name = "tenant_id")
    private String requestId;

    @NotNull
    @Column(name = "namespace")
    private String namespace;

    @Column(name = "display_name")
    private String displayName;

    @Enumerated(EnumType.STRING)
    @Column(name = "source_type")
    private SourceType sourceType;

    @Column(name = "source_location", length = 2000)
    private String sourceLocation; // URL, file path, or "inline"

    // Full OpenAPI spec stored for reference
    @Column(name = "spec_content", columnDefinition = "text")
    private String specContent;

    @Column(name = "spec_version")
    private String specVersion; // 2.0, 3.0.0, 3.1.0

    @Column(name = "spec_hash")
    private String specHash; // For change detection

    // Auth configuration
    @Column(name = "default_auth_profile_id")
    private String defaultAuthProfileId;

    // Status
    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private SourceStatus status = SourceStatus.ACTIVE;

    @Column(name = "enabled")
    private boolean enabled = true;

    // Generation metadata
    @Column(name = "tools_generated")
    private Integer toolsGenerated = 0;

    @Column(name = "last_sync_at")
    private Instant lastSyncAt;

    @Column(name = "sync_schedule")
    private String syncSchedule; // Cron expression for auto-refresh

    // Timestamps
    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;

    @Column(name = "created_by")
    private String createdBy;

    // Relationships
    @OneToMany(mappedBy = "source", cascade = CascadeType.ALL)
    private List<McpTool> tools = new ArrayList<>();

    // Getters and Setters
    public UUID getSourceId() {
        return sourceId;
    }

    public void setSourceId(UUID sourceId) {
        this.sourceId = sourceId;
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

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public SourceType getSourceType() {
        return sourceType;
    }

    public void setSourceType(SourceType sourceType) {
        this.sourceType = sourceType;
    }

    public String getSourceLocation() {
        return sourceLocation;
    }

    public void setSourceLocation(String sourceLocation) {
        this.sourceLocation = sourceLocation;
    }

    public String getSpecContent() {
        return specContent;
    }

    public void setSpecContent(String specContent) {
        this.specContent = specContent;
    }

    public String getSpecVersion() {
        return specVersion;
    }

    public void setSpecVersion(String specVersion) {
        this.specVersion = specVersion;
    }

    public String getSpecHash() {
        return specHash;
    }

    public void setSpecHash(String specHash) {
        this.specHash = specHash;
    }

    public String getDefaultAuthProfileId() {
        return defaultAuthProfileId;
    }

    public void setDefaultAuthProfileId(String defaultAuthProfileId) {
        this.defaultAuthProfileId = defaultAuthProfileId;
    }

    public SourceStatus getStatus() {
        return status;
    }

    public void setStatus(SourceStatus status) {
        this.status = status;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public Integer getToolsGenerated() {
        return toolsGenerated;
    }

    public void setToolsGenerated(Integer toolsGenerated) {
        this.toolsGenerated = toolsGenerated;
    }

    public Instant getLastSyncAt() {
        return lastSyncAt;
    }

    public void setLastSyncAt(Instant lastSyncAt) {
        this.lastSyncAt = lastSyncAt;
    }

    public String getSyncSchedule() {
        return syncSchedule;
    }

    public void setSyncSchedule(String syncSchedule) {
        this.syncSchedule = syncSchedule;
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

    public List<McpTool> getTools() {
        return tools;
    }

    public void setTools(List<McpTool> tools) {
        this.tools = tools;
    }
}