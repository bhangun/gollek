# Golek Providers

This folder contains provider implementations for cloud and local inference backends.

## Cloud Providers
* `golek-provider-openai`
* `golek-ext-cloud-gemini`
* `golek-provider-anthropic`
* `golek-provider-huggingface`
* `golek-ext-cloud-cerebras`

## Local Providers
* `golek-ext-cloud-ollama`
* `golek-provider-local`
* `golek-provider-local-vllm`
* `golek-provider-litert`

## Special Purpose
* `golek-provider-embedding`
* `golek-provider-tensorflow`

---

## How to Add a New Provider

1. Create a new module under `inference-golek/provider/`
2. Implement `LLMProvider` (and `StreamingProvider` if streaming is supported)
3. Register the provider with CDI (or factory/registry)
4. Add capability metadata (supported formats, devices, limits)
5. Add a README with supported features and configuration keys
