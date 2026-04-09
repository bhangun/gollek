# gollek-sdk-java-local

`gollek-sdk-java-local` is the single client-facing entrypoint for local Gollek SDK usage.

## Native FFI entrypoints

The module exposes GraalVM `@CEntryPoint` functions in:

- `tech.kayys.gollek.sdk.local.nativeinterop.GollekNativeEntrypoints`

Core lifecycle:

- `gollek_client_create`
- `gollek_client_destroy`
- `gollek_client_shutdown_runtime`
- `gollek_last_error`
- `gollek_clear_last_error`
- `gollek_string_free`

SDK feature entrypoints (JSON payloads):

- `gollek_create_completion_json`
- `gollek_create_completion_async_json`
- `gollek_stream_completion_json`
- `gollek_submit_async_job_json`
- `gollek_get_job_status_json`
- `gollek_wait_for_job_json`
- `gollek_batch_inference_json`
- `gollek_list_available_providers_json`
- `gollek_get_provider_info_json`
- `gollek_set_preferred_provider`
- `gollek_get_preferred_provider_json`
- `gollek_list_models_json`
- `gollek_list_models_paginated_json`
- `gollek_get_model_info_json`
- `gollek_pull_model_json`
- `gollek_delete_model`

## Native shared library build

Example:

```bash
mvn -f inference-gollek/sdk/gollek-sdk-java-local/pom.xml -DskipTests package

native-image \
  -cp inference-gollek/sdk/gollek-sdk-java-local/target/classes \
  --shared \
  -H:Name=libgollek_sdk_local \
  -H:Class=tech.kayys.gollek.sdk.local.nativeinterop.GollekNativeEntrypoints
```

All returned `char*` values must be released by calling `gollek_string_free`.
