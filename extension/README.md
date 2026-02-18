# Gollek Providers

This folder contains provider implementations for cloud and local inference backends.

## Cloud Providers
* `gollek-provider-openai`
* `gollek-ext-cloud-gemini`
* `gollek-provider-anthropic`
* `gollek-provider-huggingface`
* `gollek-ext-cloud-cerebras`

## Local Providers
* `gollek-ext-cloud-ollama`
* `gollek-provider-local`
* `gollek-provider-local-vllm`
* `gollek-provider-litert`

## Special Purpose
* `gollek-provider-embedding`
* `gollek-provider-tensorflow`

---

## How to Add a New Provider

1. Create a new module under `inference-gollek/provider/`
2. Implement `LLMProvider` (and `StreamingProvider` if streaming is supported)
3. Register the provider with CDI (or factory/registry)
4. Add capability metadata (supported formats, devices, limits)
5. Add a README with supported features and configuration keys
