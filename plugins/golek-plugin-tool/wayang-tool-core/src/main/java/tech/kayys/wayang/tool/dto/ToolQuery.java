package tech.kayys.wayang.tool.dto;

import java.util.Set;

/**
 * Tool discovery and filtering criteria
 */
public class ToolQuery {
    private String requestId;
    private String namespace;
    private Set<String> tags;
    private Set<String> capabilities;
    private CapabilityLevel maxCapabilityLevel;
    private Boolean enabled;
    private Boolean readOnly;
    private String searchTerm;
    private int page = 0;
    private int size = 50;

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

    public CapabilityLevel getMaxCapabilityLevel() {
        return maxCapabilityLevel;
    }

    public void setMaxCapabilityLevel(CapabilityLevel maxCapabilityLevel) {
        this.maxCapabilityLevel = maxCapabilityLevel;
    }

    public Boolean getEnabled() {
        return enabled;
    }

    public void setEnabled(Boolean enabled) {
        this.enabled = enabled;
    }

    public Boolean getReadOnly() {
        return readOnly;
    }

    public void setReadOnly(Boolean readOnly) {
        this.readOnly = readOnly;
    }

    public String getSearchTerm() {
        return searchTerm;
    }

    public void setSearchTerm(String searchTerm) {
        this.searchTerm = searchTerm;
    }

    public int getPage() {
        return page;
    }

    public void setPage(int page) {
        this.page = page;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

}