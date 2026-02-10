# Wayang Tool Parser

The Wayang Tool Parser module provides robust parsing and processing capabilities for various API specification formats, converting them into executable tools for the Wayang platform.

## Overview

The Tool Parser module is responsible for:
- Parsing multiple API specification formats (OpenAPI, Swagger, AsyncAPI, GraphQL, etc.)
- Converting specifications into executable MCP (Model Context Protocol) tools
- Generating appropriate schemas and validation rules
- Managing authentication and authorization mappings
- Applying guardrails and security policies

## Features

### Supported Specification Formats
- **OpenAPI 3.x**: Modern REST API specifications
- **Swagger 2.0**: Legacy API specifications  
- **AsyncAPI**: Event-driven API specifications
- **GraphQL**: GraphQL schema definitions
- **Postman Collections**: Postman API collections
- **WSDL**: SOAP web service descriptions
- **HTTP Archive (HAR)**: Recorded HTTP traffic
- **Insomnia Collections**: Insomnia API client collections

### Core Components

#### OpenAPI Parser
The OpenAPI parser handles OpenAPI 3.x specifications, extracting:
- Endpoints and HTTP methods
- Request/response schemas
- Authentication schemes
- Parameter definitions
- Example values

#### Multi-Spec Parser
Supports multiple specification formats through a unified interface:
- Automatic format detection
- Consistent parsing interface
- Format-specific optimizations

#### Schema Converter
Converts OpenAPI schemas to JSON Schema format for validation and processing.

#### Tool Generator
Transforms parsed specifications into executable MCP tools with:
- HTTP execution configuration
- Input/output schema validation
- Authentication mapping
- Guardrail enforcement

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Specs     │───▶│   Parsers        │───▶│   MCP Tools     │
│ (OpenAPI, etc.) │    │ (OpenAPI, etc.)  │    │ (Executable)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Registry       │
                       │ (Persist Tools)  │
                       └──────────────────┘
```

## Usage

### Basic Usage
```java
@Inject
OpenApiToolGenerator generator;

// Generate tools from an OpenAPI specification
GenerateToolsRequest request = new GenerateToolsRequest(
    "tenant-id",
    "namespace",
    SourceType.URL,
    "https://api.example.com/openapi.json",
    "auth-profile-id",
    "user-id",
    guardrailsConfig
);

Uni<ToolGenerationResult> result = generator.generateTools(request);
```

### Configuration Options
- **Source Types**: URL, File, Raw String, Git Repository
- **Authentication**: OAuth, API Keys, Basic Auth, Bearer Tokens
- **Guardrails**: Rate limiting, input validation, security policies
- **Capabilities**: Read-only, unsafe operations, admin functions

## Testing

The module includes comprehensive unit tests covering:
- Schema conversion functionality
- Format detection algorithms
- Parser error handling
- Tool generation workflows

Run tests with:
```bash
mvn test
```

## Integration

The Tool Parser integrates with:
- **MCP Gateway**: For tool execution
- **Memory System**: For context management
- **Guardrails Engine**: For security enforcement
- **Authentication Service**: For access control

## Best Practices

1. **Validate Specifications**: Always validate input specifications before parsing
2. **Handle Errors Gracefully**: Implement proper error handling for parsing failures
3. **Secure Endpoints**: Apply appropriate guardrails to prevent abuse
4. **Monitor Usage**: Track tool usage and performance metrics

## Contributing

Contributions to support additional specification formats are welcome. Please follow the existing patterns for parser implementation and testing.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.