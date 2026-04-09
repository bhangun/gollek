package tech.kayys.gollek.sdk.mcp;

import tech.kayys.gollek.sdk.exception.SdkException;

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

    /**
     * Lists all tools provided by a specific MCP server.
     *
     * @param name Name of the MCP server
     * @return List of tools
     * @throws SdkException if the tools could not be listed
     */
    List<McpToolModel> listTools(String name) throws SdkException;
}
