
# Complete MCP Protocol Implementation

## 📦 Module 4: MCP Client & Protocol

### Project Structure

```
inference-provider-mcp/
├── pom.xml
└── src/main/java/tech/kayys/wayang/inference/providers/mcp/
    ├── client/
    │   ├── MCPClient.java
    │   ├── MCPClientConfig.java
    │   ├── MCPConnection.java
    │   ├── MCPTransport.java
    │   ├── StdioTransport.java
    │   ├── HttpTransport.java
    │   └── WebSocketTransport.java
    ├── protocol/
    │   ├── MCPMessage.java
    │   ├── MCPRequest.java
    │   ├── MCPResponse.java
    │   ├── MCPNotification.java
    │   ├── MCPError.java
    │   └── JsonRpcMessage.java
    ├── server/
    │   ├── MCPServerRegistry.java
    │   ├── MCPServerDescriptor.java
    │   └── MCPServerConnection.java
    ├── tools/
    │   ├── MCPTool.java
    │   ├── MCPToolExecutor.java
    │   ├── MCPToolRegistry.java
    │   └── MCPToolResult.java
    ├── resources/
    │   ├── MCPResourceProvider.java
    │   ├── MCPResource.java
    │   └── MCPResourceContent.java
    ├── prompts/
    │   ├── MCPPromptProvider.java
    │   └── MCPPrompt.java
    ├── provider/
    │   ├── MCPProvider.java
    │   ├── MCPProviderFactory.java
    │   └── RemoteMCPProvider.java
    └── memory/
        ├── MCPMemoryProvider.java
        └── MCPMemoryStore.java
```
