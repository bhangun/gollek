# Gollek SDK :: NLP Pipelines

The `gollek-sdk-nlp` module bridges the gap between text generation models running on the engine and tokenization workflows. 
It aggregates Tokenizer strategies into high-level pipelines.

## Pipelines
- **TextGenerationPipeline**: Sends prompt sequences and generation configurations to the low-level inference engine via an easy-to-use API string format.
- **TokenizerPipeline**: Hooks into `gollek-tokenizer-core` to provide BPE encoding/decoding strategies explicitly attached to loaded vocabularies (e.g. `tokenizer.json`).

## Example

```java
import tech.kayys.gollek.ml.nlp.TextGenerationPipeline;
import tech.kayys.gollek.ml.nlp.TokenizerPipeline;

var tokenizer = new TokenizerPipeline("Qwen/Qwen2.5-0.5B", "/path/to/tokenizer.json");
var pipeline = new TextGenerationPipeline("Qwen/Qwen2.5", tokenizer);

String output = pipeline.process("Explain quantum mechanics simply.");
```
