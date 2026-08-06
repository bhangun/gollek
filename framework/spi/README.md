# Gollek SPI (Service Provider Interfaces)

Centralized location for all Gollek Service Provider Interfaces (SPIs).

## 📦 Module Structure

```
inference-gollek/spi/
├── pom.xml                       # Parent POM
├── gollek-spi-plugin/            # Plugin lifecycle and extension points
├── gollek-spi-model/             # Model loading and metadata
├── gollek-spi-inference/         # Inference execution contracts
└── gollek-spi-provider/          # Provider discovery and management
```

## 🎯 Purpose

Each SPI module defines a specific contract:

### gollek-spi-plugin
- Plugin lifecycle management
- Extension points
- Plugin registry
- Plugin health monitoring
- Plugin state management

### gollek-spi-model  
- Model loading interface
- Model metadata
- Model format abstraction
- Model repository contracts
- Model routing

### gollek-spi-inference
- Inference session management
- Input/output processing
- Execution contracts
- Result handling
- Device management

### gollek-spi-provider
- Provider discovery
- Provider configuration
- Backend selection
- Provider lifecycle
- Provider routing
- Backend-neutral feature pipelines through `ModelPipeline`
- Global extension adapters through `FeatureAdapter`

## Feature Pipelines

`gollek-spi-provider` also contains the feature pipeline SPI:

- `ModelPipeline` is implemented by OCR, TTS, ASR, diffusion, evaluation, or other model-specific orchestration projects.
- `ModelPipelineRequest` preserves the original `ProviderRequest` and adds neutral model facts such as `format`, `provider`, and backend-specific inspection keys.
- `ModelPipelineRegistry` discovers feature jars through CDI or Java `ServiceLoader`.

This layer is intentionally not ONNX-only. ONNX, LiteRT, safetensor, GGUF, or future runners can all publish inspected facts into the same request envelope, then delegate to a matching pipeline when a model needs preprocessing, multiple sessions, tokenizer/audio/image postprocessing, or other orchestration above a single backend call. Current provider facts include `format=onnx`, `format=litert`, and `format=safetensors`.

External extensions should live outside the core engine when they are experimental, domain-specific, or maintained by a separate team. See `features/gollek-paddleocr` for a minimal out-of-core example.

## Feature Adapters

`FeatureAdapter` is the generic SPI inside an extension. Use it for training, optimization, quantization, backend bridges, model conversion/export, dataset tooling, evaluation, or other domain-specific capabilities that should live outside Gollek core.

Adapters can be declared through Java `ServiceLoader`:

```text
META-INF/services/tech.kayys.gollek.spi.feature.FeatureAdapter
```

Extension manifests can describe all extension surfaces in one schema:

```json
{
  "schema_version": 1,
  "id": "gollek-awq-lab",
  "name": "Gollek AWQ Lab",
  "version": "0.1.0",
  "requires": {
    "gollek_spi": ">=0.1.0"
  },
  "capabilities": [
    {
      "id": "awq-quantizer",
      "kind": "quantization",
      "class": "example.AwqQuantizer",
      "formats": ["safetensors", "gguf"],
      "inputs": ["model"],
      "outputs": ["model"],
      "targets": ["llm"]
    },
    {
      "id": "metal-backend-profile",
      "kind": "backend",
      "provider": "metal"
    }
  ]
}
```

The preferred manifest path is `META-INF/gollek-extension.json`; Gollek still reads the legacy `META-INF/gollek-feature.json` path for compatibility. The legacy `pipelines` array is still supported and is treated as `kind: "pipeline"`. New manifests may also use top-level `adapters` for adapter-only declarations. The CLI scans all three arrays: `pipelines`, `capabilities`, and `adapters`.

Runtime extension jars and distributable extension zip packages can be managed with the CLI:

```bash
gollek extensions kinds
gollek extensions init features/gollek-my-extension --pipeline-id my-feature --gollek-path gollek
gollek extensions init features/gollek-awq-lab --kind quantization --adapter-id awq-quantizer --input model --output model --target llm --format safetensors,gguf --gollek-path gollek
gollek extensions --plugin-dir path/to/extensions --details --paths
gollek extensions inspect path/to/extension.jar --strict
gollek extensions install path/to/extension.jar
gollek extensions inspect path/to/gollek-extension-package.zip --strict
gollek extensions install path/to/gollek-extension-package.zip
gollek extensions index --output gollek-extensions.index.json
gollek extensions doctor --strict --json
gollek extensions doctor --repair --dry-run
gollek extensions doctor --plugin-dir path/to/extensions --strict
gollek extensions backups --json
gollek extensions backups extension-id --prune --keep 5 --dry-run
gollek extensions rollback extension-id --dry-run
gollek extensions remove extension-id
gollek extensions --doctor --paths
gollek run --model <model> --pipeline my-feature --prompt "hello"
```

`features` remains an alias for `extensions` during migration. `inspect --strict` validates service providers, manifest pipeline classes, duplicate pipeline ids, and modality names before install. When given a package zip, `inspect` reads `gollek-package.json`, extracts the declared jar artifact to a temporary file, applies the same jar checks, and compares package descriptor id/name/version/requirements plus artifact manifest metadata against the jar manifest. Package installs reject descriptor-to-jar mismatches before copying. `extensions doctor --json` applies the same checks across every configured extension path, reports validation summary counts, and returns a non-zero exit code for CI when errors are found. Add `--strict` to treat warnings as failures. The root `extensions --doctor --json` flag remains useful when you want discovery output and doctor details in one report.

`extensions kinds` lists supported scaffold kinds, generated service contracts, default inputs/outputs, target hints, and copyable examples. `extensions init` defaults to `--kind pipeline` and generates a `ModelPipeline` service. Non-pipeline kinds such as `training`, `optimization`, `quantization`, `backend`, `exporter`, `dataset`, and `evaluator` generate a `FeatureAdapter` service plus a manifest capability so global extension work can start outside the core engine. Generated manifests include service metadata and an ahead-of-time loading note.

`extensions install` accepts either a jar or a package zip. Zip installs read the package descriptor, extract the declared jar artifact, and install that jar into the target extension directory. The command maintains a `gollek-extensions.lock.json` file in the target extension directory. The lock stores each installed jar's filename, path, SHA-256, size, install time, runtime version, providers, adapter providers, extension id, capability ids, capability kinds, and pipeline ids. Package zip installs also record `source_package` provenance with the zip path, package SHA-256, package size, descriptor entry, jar artifact entry, jar artifact SHA-256, jar artifact size, package id, version, kind, install filename, and required runtime metadata. `extensions doctor` compares current jars against that lock so manual replacement, stale lock entries, and missing installed jars are visible during validation; when a recorded source package zip still exists, doctor also verifies its size, SHA-256, descriptor entry, jar artifact entry, jar artifact fingerprint, package id/version/kind, and recorded requirements. Existing jars without lock entries remain valid; they are simply untracked until reinstalled through the CLI. `extensions doctor --repair --dry-run` previews lock changes, and `extensions doctor --repair` refreshes lock entries for existing readable jars, enriches older package provenance when the recorded package artifact still matches the installed jar, and removes stale lock entries from scanned directories.

`extensions doctor` also detects collisions across extension ids, feature ids, pipeline ids, adapter ids, provider classes, and implementation classes. Collisions remain warnings by default and become failures with `--strict`.

JVM runtimes can dynamically load extension jars from `~/.gollek/extensions`, `GOLLEK_EXTENSION_PATH`, or `-Dgollek.extensions.path`. GraalVM native image runtimes are AOT-only: dynamic jar classloading is disabled, so extension pipelines/adapters must be present on the build classpath and registered before native compilation. Installed jars are still manifest-inspectable for doctor, migration, and inventory. Use `extensions index --output gollek-extensions.index.json` to emit jar checksums, ServiceLoader entries, manifest capabilities, validation status, and native-image hints for launchers or build scripts.

`extensions migrate --dry-run` previews copying legacy `~/.gollek/features` jars into the runtime `~/.gollek/extensions` directory. Running without `--dry-run` copies readable jars and writes `gollek-extensions.lock.json`. Pass `--force` when a target jar already exists and should be replaced after a rollback backup is created.

`extensions finalize-migration --dry-run` previews legacy jars that are already represented by runtime extension jars. Running without `--dry-run` archives each matched legacy jar into rollback backups and removes it from `~/.gollek/features`. `extensions prune-legacy` remains available as the lower-level cleanup command.

Forced installs and removals create rollback backups under `.gollek-extension-backups` by default. `extensions rollback <jar-or-extension>` restores the newest matching backup, writes a fresh install lock, and saves the current jar first so rollback itself can be undone. Pass `--no-backup` to `install --force` or `remove` when that safety copy is not desired.

`extensions backups` lists rollback history with extension metadata, reason, size, checksum, source path, and metadata path. Add a target to filter by jar filename, extension id, feature id, pipeline id, or provider class. `extensions backups <target> --prune --keep 5 --dry-run` previews cleanup, and dropping `--dry-run` deletes older backup jars plus their `.json` sidecars.

`extensions --dir path/to/extensions` and `extensions --plugin-dir path/to/extensions` are manifest scanners for extension directories or individual jars. They list declared features/adapters without classloading the jar, which is useful for review and CI staging. To make a scanned pipeline active for inference, install the jar into `~/.gollek/extensions` or launch Gollek with `GOLLEK_EXTENSION_PATH` / `-Dgollek.extensions.path`. Legacy `~/.gollek/features` is only a migration source. General plugin lifecycle and `~/.gollek/plugins` loading still live in `gollek/core/plugin/gollek-plugin-core`.

Legacy path configuration such as `GOLLEK_FEATURE_PATH` or `-Dgollek.features.path` is ignored. `extensions doctor` reports those settings as warnings, and `--strict` fails when they are still present.

`extensions doctor --json` includes a compact `legacy_migration` block. It keeps legacy jars out of runtime discovery while still showing whether `migrate --dry-run` or `finalize-migration --dry-run` would clean up old installs.

Extension manifests should declare runtime compatibility:

```json
{
  "schema_version": 1,
  "requires": {
    "gollek_spi": "0.1.0-SNAPSHOT",
    "java": ">=25"
  }
}
```

`extensions inspect` and `extensions doctor` compare `requires.gollek`, `requires.gollek_runtime`, `requires.gollek_cli`, or `requires.gollek_spi` against the active Gollek runtime. Exact versions, wildcard forms such as `0.1.x`, and simple constraints such as `>=0.1.0,<0.2.0` are supported. Incompatible versions are reported as errors; missing compatibility metadata is reported as a warning.

Extension jars are installed into `~/.gollek/extensions` by default, or loaded from custom paths with `GOLLEK_EXTENSION_PATH` / `-Dgollek.extensions.path`. Legacy feature paths are still supported for existing installations.

## 🔧 Usage

Add dependency to your module:

```xml
<!-- Plugin SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-plugin</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Model SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-model</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Inference SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-inference</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Provider SPI -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-spi-provider</artifactId>
    <version>${project.version}</version>
</dependency>
```

## 🏗️ Architecture Principles

The SPI modules follow these principles:

1. **Interface-only**: Define contracts, not implementations
2. **Independent**: Each SPI can evolve independently  
3. **Minimal dependencies**: Only depend on what's necessary
4. **Stable contracts**: Backward-compatible changes only
5. **Well-documented**: Clear JavaDoc for all interfaces
6. **Centralized location**: All SPIs in one place for easy discovery

## 📝 Implementation Guide

To implement a SPI:

1. Add the SPI dependency to your module
2. Implement the interface(s)
3. Register implementation via `META-INF/services`
4. Use appropriate annotations

Example:

```java
// Implement Model SPI
public class MyModelLoader implements ModelLoader {
    @Override
    public Model load(Path path) {
        // Implementation
    }
}

// Register in META-INF/services
// META-INF/services/tech.kayys.alkhawarizm.spi.model.ModelLoader
tech.kayys.alkhawarizm.spi.model.MyModelLoader
```

## 📁 Location

**Old Structure** (deprecated):
```
inference-gollek/core/
├── gollek-spi-plugin/
├── gollek-spi-model/
├── gollek-spi-inference/
└── gollek-spi-provider/
```

**New Structure** (current):
```
inference-gollek/spi/
├── gollek-spi-plugin/
├── gollek-spi-model/
├── gollek-spi-inference/
└── gollek-spi-provider/
```

## 🔗 Related Modules

### Implementations (in `core/`)
- `gollek-plugin-core` - Plugin implementation
- `gollek-model-registry` - Model registry implementation
- `gollek-provider-core` - Provider implementation

### Runtime (in `runtime/`)
- `gollek-runtime` - Main runtime
- `gollek-engine` - Engine implementation

## 🚀 Benefits of Centralized SPI Location

1. **Easy Discovery**: All SPIs in one place
2. **Clear Separation**: SPIs separate from implementations
3. **Better Organization**: Logical grouping
4. **Simplified Dependencies**: Single parent for all SPIs
5. **Independent Evolution**: SPIs can change without affecting core

---

**Version**: 1.0.0-SNAPSHOT  
**Parent**: gollek-parent  
**Location**: `inference-gollek/spi/`
