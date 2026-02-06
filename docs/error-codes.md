# Golek Error Codes

Generated from `ErrorCode` at build time.

## MODEL

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| MODEL_001 | 404 | false | Model not found |
| MODEL_002 | 404 | false | Model version not found |
| MODEL_003 | 400 | false | Invalid model format |
| MODEL_004 | 400 | false | Model file corrupted |
| MODEL_005 | 400 | false | Model exceeds size limit |
| MODEL_006 | 403 | false | Model signature verification failed |
| MODEL_007 | 400 | false | Model not compatible with selected runner |

## TENSOR

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| TENSOR_001 | 400 | false | Tensor shape mismatch |
| TENSOR_002 | 400 | false | Tensor data type mismatch |
| TENSOR_003 | 400 | false | Invalid tensor data |
| TENSOR_004 | 400 | false | Tensor size does not match shape |
| TENSOR_005 | 500 | true | Tensor conversion failed |
| TENSOR_006 | 400 | false | Required input tensor missing |

## DEVICE

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| DEVICE_001 | 503 | true | Requested device not available |
| DEVICE_002 | 503 | true | Device out of memory |
| DEVICE_003 | 500 | false | Device initialization failed |
| DEVICE_004 | 500 | true | Device driver error |
| DEVICE_005 | 503 | false | GPU not found |
| DEVICE_006 | 503 | false | TPU not available |
| DEVICE_007 | 501 | false | NPU not supported on this platform |

## QUOTA

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| QUOTA_001 | 429 | false | Quota exceeded |
| QUOTA_002 | 429 | false | Rate limit exceeded |
| QUOTA_003 | 429 | true | Too many concurrent requests |
| QUOTA_004 | 429 | false | Storage quota exceeded |
| QUOTA_005 | 429 | true | Compute quota exceeded |

## AUTH

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| AUTH_001 | 401 | false | Invalid authentication token |
| AUTH_002 | 401 | false | Authentication token expired |
| AUTH_003 | 401 | false | Tenant not found |
| AUTH_004 | 403 | false | Permission denied |
| AUTH_005 | 403 | false | Tenant account suspended |

## INIT

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| INIT_001 | 500 | false | Runner initialization failed |
| INIT_002 | 500 | true | Model loading failed |
| INIT_003 | 500 | false | Native library loading failed |
| INIT_004 | 500 | false | Invalid configuration |
| INIT_005 | 500 | false | Required dependency missing |

## RUNTIME

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| RUNTIME_001 | 500 | true | Inference execution failed |
| RUNTIME_002 | 504 | true | Inference request timeout |
| RUNTIME_003 | 500 | true | Out of memory during inference |
| RUNTIME_004 | 500 | true | Native library crashed |
| RUNTIME_005 | 500 | false | Invalid runner state |
| RUNTIME_006 | 400 | false | Batch size exceeds limit |

## STORAGE

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| STORAGE_001 | 500 | true | Failed to read from storage |
| STORAGE_002 | 500 | true | Failed to write to storage |
| STORAGE_003 | 404 | false | Storage resource not found |
| STORAGE_004 | 503 | true | Storage connection failed |
| STORAGE_005 | 403 | false | Storage permission denied |

## CONVERSION

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| CONVERSION_001 | 500 | true | Model conversion failed |
| CONVERSION_002 | 400 | false | Target format not supported |
| CONVERSION_003 | 504 | true | Model conversion timeout |
| CONVERSION_004 | 500 | false | Converted model validation failed |
| CONVERSION_005 | 500 | true | Model quantization failed |

## VALIDATION

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| VALIDATION_001 | 400 | false | Required field missing |
| VALIDATION_002 | 400 | false | Invalid field format |
| VALIDATION_003 | 400 | false | Validation constraint violated |

## CIRCUIT

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| CIRCUIT_001 | 503 | true | Circuit breaker open |
| CIRCUIT_002 | 503 | true | All runner attempts failed |
| CIRCUIT_003 | 503 | true | Fallback execution failed |

## PROVIDER

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| PROVIDER_001 | 503 | true | Provider not initialized |
| PROVIDER_002 | 503 | true | Provider unavailable |
| PROVIDER_003 | 504 | true | Provider timeout |
| PROVIDER_004 | 401 | false | Provider authentication failed |
| PROVIDER_005 | 429 | true | Provider rate limit exceeded |
| PROVIDER_006 | 429 | false | Provider quota exceeded |
| PROVIDER_007 | 502 | true | Provider returned invalid response |
| PROVIDER_008 | 502 | true | Provider stream failed |
| PROVIDER_009 | 400 | false | Provider request invalid |
| PROVIDER_010 | 500 | false | Provider initialization failed |

## ROUTING

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| ROUTING_001 | 503 | true | No compatible provider available |
| ROUTING_002 | 404 | false | Provider not found |
| ROUTING_003 | 403 | false | Routing policy rejected request |

## PLUGIN

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| PLUGIN_001 | 500 | false | Plugin initialization failed |
| PLUGIN_002 | 500 | true | Plugin execution failed |
| PLUGIN_003 | 500 | false | Plugin configuration invalid |

## CONFIG

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| CONFIG_001 | 500 | false | Required configuration missing |
| CONFIG_002 | 500 | false | Invalid configuration value |
| CONFIG_003 | 400 | false | Unsupported configuration |

## NETWORK

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| NETWORK_001 | 504 | true | Network timeout |
| NETWORK_002 | 503 | true | Network unreachable |
| NETWORK_003 | 503 | true | DNS resolution failed |
| NETWORK_004 | 502 | false | TLS handshake failed |
| NETWORK_005 | 502 | true | Network protocol error |
| NETWORK_006 | 502 | true | Network response invalid |

## STREAM

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| STREAM_001 | 500 | false | Stream initialization failed |
| STREAM_002 | 502 | true | Stream disconnected |
| STREAM_003 | 502 | true | Stream protocol error |
| STREAM_004 | 504 | true | Stream timeout |

## INTERNAL

| Code | HTTP | Retryable | Message |
| --- | --- | --- | --- |
| INTERNAL_001 | 500 | true | Internal server error |
| INTERNAL_002 | 500 | true | Database error |
| INTERNAL_003 | 500 | true | Cache error |
| INTERNAL_004 | 500 | true | Serialization error |

