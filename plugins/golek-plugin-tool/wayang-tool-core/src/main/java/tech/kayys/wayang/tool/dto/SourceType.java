package tech.kayys.wayang.tool.dto;

public enum SourceType {
    // OpenAPI 3.x
    OPENAPI_3_URL,
    OPENAPI_3_FILE,
    OPENAPI_3_RAW,
    
    // Swagger 2.0
    SWAGGER_2_URL,
    SWAGGER_2_FILE,
    SWAGGER_2_RAW,
    
    // Postman Collection
    POSTMAN_URL,
    POSTMAN_FILE,
    POSTMAN_RAW,
    
    // AsyncAPI
    ASYNCAPI_URL,
    ASYNCAPI_FILE,
    ASYNCAPI_RAW,
    
    // GraphQL
    GRAPHQL_URL,
    GRAPHQL_FILE,
    GRAPHQL_RAW,
    
    // WSDL
    WSDL_URL,
    WSDL_FILE,
    
    // Generic (auto-detect)
    URL,
    FILE,
    RAW,
    GIT
}