# Model Storage Service

The ModelStorageService provides a unified interface for storing and retrieving model files across multiple storage backends.

## Features

- **Multi-Backend Support**: Supports AWS S3, Google Cloud Storage, Azure Blob Storage, and local filesystem
- **Panache Repository Pattern**: Implements Quarkus Panache repository pattern for database operations
- **Configuration Driven**: Fully configurable through application properties
- **Reactive Programming**: Uses Mutiny for reactive operations
- **Comprehensive Error Handling**: Proper validation and error handling

## Storage Providers

### AWS S3
- Configurable bucket, region, and credentials
- S3-compatible services (like MinIO) support
- Proper AWS SDK integration

### Google Cloud Storage
- Full GCS integration with Google Cloud SDK
- Configurable bucket and project settings

### Azure Blob Storage
- Complete Azure Blob Storage implementation
- Connection string-based authentication

### Local Storage
- Filesystem-based storage for development
- Configurable base path

## Repository Pattern

The ModelRepository implements the Panache repository pattern with custom finder methods:
- `findByTenantAndModelId()` - Find model by tenant and model ID
- `findByTenant()` - Find models by tenant
- `findByStage()` - Find models by stage
- `findById()` - Find model by ID

## Configuration

Configure the storage provider in your application.properties:

```properties
# Storage provider (local, s3, gcs, azure)
inference.model-storage.provider=local

# Local storage
inference.model-storage.local.base-path=/var/lib/inference/models

# S3 configuration
inference.model-storage.s3.bucket=ml-models
inference.model-storage.s3.region=us-east-1
inference.model-storage.s3.path-prefix=models/
inference.model-storage.s3.access-key-id=your-access-key
inference.model-storage.s3.secret-access-key=your-secret-key
inference.model-storage.s3.endpoint=https://s3.amazonaws.com  # For custom S3-compatible services

# GCS configuration
inference.model-storage.gcs.bucket=ml-models
inference.model-storage.gcs.project-id=your-project-id

# Azure configuration
inference.model-storage.azure.container=ml-models
inference.model-storage.azure.connection-string=your-connection-string
```

## Usage

The service provides the following operations:
- `uploadModel()` - Upload model to storage
- `downloadModel()` - Download model from storage
- `deleteModel()` - Delete model from storage
- `modelExists()` - Check if model exists in storage