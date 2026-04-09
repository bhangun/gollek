# Gollek SDK :: Model Hub Integration

The `gollek-sdk-hub` connects local storage mechanisms and serialization logic to the runtime layers.

## Responsibilities
- Maps weights from binary files (like `.safetensors`) directly into off-heap structured memory via `gollek-runtime-tensor`.
- Organizes the `state_dict` loading mechanism, translating tensor arrays onto initialized `Module`s in `gollek-sdk-nn`.
- Uses `slf4j` for observability of weight ingestion performance.

## Core Flow
When `ModelHub.loadWeights(Path path, Module module)` is called:
1. Locates all `.safetensors` files within the provided path.
2. Cross-references internal tensor names against the `Parameter` properties of the `Module`.
3. Calls the underlying memory pools to deserialize the structure explicitly into the unified memory subsystem.
