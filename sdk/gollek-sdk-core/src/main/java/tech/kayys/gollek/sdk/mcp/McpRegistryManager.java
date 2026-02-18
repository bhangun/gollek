package tech.kayys.gollek.sdk.core.mcp;

import tech.kayys.gollek.sdk.core.exception.SdkException;

import java.util.List;

public interface McpRegistryManager {

    String registryPath();

    List<String> add(McpAddRequest request) throws SdkException;

    McpServerView show(String name) throws SdkException;

    List<McpServerSummary> list() throws SdkException;

    void remove(String name) throws SdkException;

    void rename(String oldName, String newName) throws SdkException;

    void edit(McpEditRequest request) throws SdkException;

    void setEnabled(String name, boolean enabled) throws SdkException;

    int importFromFile(String filePath, boolean replace) throws SdkException;

    int exportToFile(String filePath, String name) throws SdkException;

    McpDoctorReport doctor() throws SdkException;

    McpTestReport test(String name, boolean all, long timeoutMs) throws SdkException;
}

