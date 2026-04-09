# Gollek SDK :: Data Loaders

The `gollek-sdk-data` module is responsible for bridging incoming data types to memory-efficient `MultiTensor` constructs compatible with the inference engines.

## Key Features
- **Dataset Abstraction**: Implements `Dataset<T>` maps and utilities to lazy-load training/inference data.
- **Batched Iteration**: `DataLoader` clusters heterogeneous input records into memory-contiguous `DType`/`Device` tensors.
- **Data Transforms**: Functional map pipelines that pre-process data before passing it into `gollek-sdk-autograd`.

## Typical Workflow
In inference setups, input strings or vectors bypass the `DataLoader` directly to the `Pipeline`. But for training regimens, `gollek-sdk-data` allows:
```java
Dataset<InputStruct> ds = new MultiModalDataset("/opt/data");
DataLoader loader = DataLoader.builder(ds).batchSize(64).shuffle(true).build();

for(Batch batch : loader) {
    // Process continuous tensor mappings in memory
}
```
