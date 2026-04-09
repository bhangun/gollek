# Gollek API REST Module

This module provides REST API endpoints for the Gollek inference platform, offering a unified interface for model inference, management, and monitoring operations.

## Features

- **Model Inference API**: REST endpoints for synchronous and asynchronous model inference
- **Model Management API**: Administrative endpoints for model lifecycle management
- **Health Monitoring**: Comprehensive health check endpoints
- **Multi-tenancy Support**: Built-in tenant isolation and context management
- **Fault Tolerance**: Circuit breaker, retry, and bulkhead patterns
- **Streaming Support**: Server-sent events for streaming inference results

## API Endpoints

### Inference API (`/v1/infer`)

- `POST /completions` - Synchronous model inference
- `POST /async` - Submit asynchronous inference jobs
- `POST /stream` - Streaming inference results
- `GET /async/{jobId}` - Get async job status
- `POST /batch` - Batch inference processing
- `DELETE /{requestId}` - Cancel inference request

### Model Management API (`/v1/models`)

- `GET /` - List all models
- `GET /{modelId}` - Get model details
- `POST /` - Register new model
- `PUT /{modelId}` - Update model
- `DELETE /{modelId}` - Delete model
- `POST /{modelId}/warmup` - Warmup model

### Health API (`/health`)

- `GET /` - Standard health check
- `GET /detailed` - Detailed health status
- `GET /status` - Simple health status

## Configuration

The API module supports the following configuration properties:

```properties
# Multitenancy settings
gollek.multitenancy.enabled=false

# Inference timeouts
gollek.inference.timeout=30S
gollek.inference.max-retries=2

# Fault tolerance settings
gollek.fault-tolerance.bulkhead-size=100
gollek.fault-tolerance.queue-size=50
```

## Security

The API implements JWT-based authentication and role-based authorization:

- `USER` role: Access to inference endpoints
- `ADMIN` role: Access to all endpoints
- `MODEL_MANAGER` role: Access to model management endpoints

## Error Handling

The API follows standard HTTP status codes and provides detailed error responses:

- `200 OK`: Successful requests
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side errors

## Architecture

The module follows a layered architecture:

```
REST Controllers
    ↓
Business Services
    ↓
Data Access Layer
    ↓
External Services
```

## Dependencies

- Quarkus REST
- MicroProfile OpenAPI
- MicroProfile JWT
- MicroProfile Fault Tolerance
- MicroProfile Health
- Micrometer Metrics

## Development

To run tests:

```bash
mvn test
```

To build the module:

```bash
mvn clean install
```

## Future Enhancements

- Enhanced rate limiting strategies
- More granular permission controls
- Advanced caching mechanisms
- GraphQL API support
- Webhook notifications