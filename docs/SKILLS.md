# Gollek Native Build — Skill Knowledge Item

## Summary

Gollek is a Java 25 / Quarkus 3.32 inference engine that compiles to a GraalVM native image for macOS/arm64 (Apple Silicon). This KI documents every build configuration rule needed to produce a working native binary, common failure modes, and the exact fixes.

---

## Environment

| Dimension | Value |
|-----------|-------|
| Language  | Java 25 (preview features + `jdk.incubator.vector`) |
| Framework | Quarkus 3.32.2 |
| GraalVM   | GraalVM 25 (`graalvm-25.jdk`) |
| Arch      | macOS arm64 (Apple Silicon / Metal) |
| Build cmd | `./install-local.sh -n` |
| Output    | `ui/gollek-cli/target/gollek` (293 MB Mach-O arm64) |
| Installed | `~/.local/bin/gollek` |
| JDK req.  | `[25, 27)` — enforced by maven-enforcer-plugin |

---

## Build Entry Point

```bash
# Full reactor native build
./install-local.sh -n > build_native.log 2>&1

# Resume after fixing a single module
mvn -pl ui/gollek-cli -am clean install -Dnative -DskipTests

# Read the error (filter noise)
grep -E "(Fatal error|UnsupportedFeature|Discovered unresolved|was found in the image heap)" build_native.log
tail -80 build_native.log
```

---

## Working `application.properties` (Native Args)

File: [`ui/gollek-cli/src/main/resources/application.properties`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/ui/gollek-cli/src/main/resources/application.properties)

```properties
# ─── GraalVM Native Image Arguments ───────────────────────────────────────────
quarkus.native.additional-build-args[0]=--enable-native-access=ALL-UNNAMED
quarkus.native.additional-build-args[1]=-H:+ReportExceptionStackTraces
quarkus.native.additional-build-args[2]=-H:+AddAllCharsets
quarkus.native.additional-build-args[3]=-H:+UnlockExperimentalVMOptions
quarkus.native.additional-build-args[4]=-H:+VectorAPISupport
# JLine Kernel32 Windows stubs (arm64 mac build still needs these)
quarkus.native.additional-build-args[5]=--initialize-at-run-time=org.jline.nativ.Kernel32$MENU_EVENT_RECORD
quarkus.native.additional-build-args[6]=--initialize-at-run-time=org.jline.nativ.Kernel32$FOCUS_EVENT_RECORD
quarkus.native.additional-build-args[7]=--initialize-at-run-time=org.jline.nativ.Kernel32$INPUT_RECORD
quarkus.native.additional-build-args[8]=--initialize-at-run-time=org.jline.nativ.Kernel32$COORD
quarkus.native.additional-build-args[9]=--initialize-at-run-time=org.jline.nativ.Kernel32$KEY_EVENT_RECORD
quarkus.native.additional-build-args[10]=--initialize-at-run-time=org.jline.nativ.Kernel32$MOUSE_EVENT_RECORD
quarkus.native.additional-build-args[11]=--initialize-at-run-time=org.jline.nativ.Kernel32$WINDOW_BUFFER_SIZE_RECORD
quarkus.native.additional-build-args[12]=--initialize-at-run-time=org.jline.nativ.Kernel32$CONSOLE_SCREEN_BUFFER_INFO
quarkus.native.additional-build-args[13]=--initialize-at-run-time=org.jline.nativ.Kernel32$SMALL_RECT
quarkus.native.additional-build-args[14]=--initialize-at-run-time=org.jline.nativ.Kernel32$CHAR_INFO
quarkus.native.additional-build-args[15]=--initialize-at-run-time=org.jline.nativ.Kernel32
# LibTorch FFM binding uses Cleaner at static init
quarkus.native.additional-build-args[16]=--initialize-at-run-time=tech.kayys.gollek.inference.libtorch.core.TorchTensor
quarkus.native.additional-build-args[17]=--initialize-at-run-time=tech.kayys.gollek.inference.libtorch.core.TorchTensor$CleanerHolder
# Disable SharedArena (incompatible with jdk.incubator.vector in GraalVM 25)
quarkus.native.additional-build-args[18]=-H:-SharedArenaSupport
# LlamaCpp JNI stubs
quarkus.native.additional-build-args[19]=--initialize-at-run-time=tech.kayys.gollek.llama
quarkus.native.additional-build-args[20]=--initialize-at-run-time=tech.kayys.gollek.inference.llamacpp
# AccelTensor uses Arena.ofAuto at field init (FFM off-heap)
quarkus.native.additional-build-args[21]=--initialize-at-run-time=tech.kayys.gollek.safetensor.core.tensor.AccelTensor
# Java preview + incubator modules
quarkus.native.additional-build-args[22]=--enable-preview
quarkus.native.additional-build-args[23]=-J--enable-preview
quarkus.native.additional-build-args[24]=--add-modules=jdk.incubator.vector
quarkus.native.additional-build-args[25]=-J--add-modules=jdk.incubator.vector
# Debug: trace any remaining Cleaner instantiations
quarkus.native.additional-build-args[26]=--trace-object-instantiation=java.lang.ref.Cleaner
quarkus.native.native-image-xmx=12g
```

---

## Known Failure Modes & Fixes

### Failure 1: `Discovered unresolved type: jakarta.enterprise.invoke.Invoker`

**Class**: `io.quarkus.vertx.runtime.VertxEventBusConsumerRecorder`  
**Root cause**: CDI API pinned to 4.0.1 in project POMs; Quarkus 3.32+ requires CDI 4.1.0.

**Locations that had the bad pin (now fixed):**
- [`pom.xml`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/pom.xml) — parent `dependencyManagement`
- [`bom/pom.xml`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/bom/pom.xml) — gollek-bom `dependencyManagement`
- [`ui/gollek-cli/pom.xml`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/ui/gollek-cli/pom.xml) — explicit `<dependency>` declaration

**Fix**: Remove all three pins. The Quarkus BOM (`io.quarkus.platform:quarkus-bom:3.32.2`) manages CDI, inject, and validation versions. Never add explicit versions for these in a Quarkus project.

> [!IMPORTANT]
> **Rule**: In any Quarkus project, do **not** add `<version>` to `jakarta.enterprise.cdi-api`, `jakarta.inject-api`, or `jakarta.validation-api` in `dependencyManagement`. Always delegate to the Quarkus BOM.

**Verify the resolved version:**
```bash
mvn dependency:tree -pl ui/gollek-cli | grep cdi-api
# Must show: jakarta.enterprise:jakarta.enterprise.cdi-api:jar:4.1.0:compile
```

---

### Failure 2: `_Bean` Object Found in Image Heap

**Error pattern**:
```
An object of type 'tech.kayys.gollek.safetensor.loader.SafetensorLoadCache_Bean'
was found in the image heap. This type is marked for initialization at run time.
```

**Root cause**: A package-level `--initialize-at-run-time=tech.kayys.gollek.safetensor.loader` rule catches Quarkus-generated `_Bean` classes for CDI beans in that package. Quarkus bakes these beans into the image heap at build time, but the rule marks them as runtime-init — a contradiction.

**Fix**: **Never** use package-level `--initialize-at-run-time` for packages that contain CDI `@ApplicationScoped`/`@RequestScoped`/`@Singleton` beans. Use class-level rules targeting only non-CDI classes that instantiate `Cleaner`, `Arena.ofShared()`, or JNI resources in `<clinit>`.

**Pattern for diagnosing the next occurrence:**
```bash
grep "was found in the image heap" build_native.log
# Then add ONLY the specific non-Bean class to the run-time init list
# or use --initialize-at-build-time for the _Bean classes instead
```

---

### Failure 3: `Arena.ofShared()` in GraalVM 25

**Error pattern**: Build fails with reference to `SharedArena` or `Cleaner` during heap scan.  
**Fix**: `quarkus.native.additional-build-args[18]=-H:-SharedArenaSupport`

Also ensure all code uses `Arena.ofAuto()` instead of `Arena.ofShared()`:
```java
// BAD (will fail native-image)
Arena arena = Arena.ofShared();

// GOOD
Arena arena = Arena.ofAuto();
```

---

### Failure 4: Vector API with GraalVM

**Error**: `VectorAPISupport requires UnlockExperimentalVMOptions`  
**Fix**: Enable both flags together (indices 3 and 4 above).

Also required in the JVM build phase (parent `pom.xml` compiler args):
```xml
<compilerArgs>
    <arg>--enable-preview</arg>
    <arg>--add-modules</arg>
    <arg>jdk.incubator.vector</arg>
</compilerArgs>
```

---

## Dependency Management Rules

```
Priority (highest → lowest):
  module pom.xml dependencyManagement
  → imported BOM (gollek-bom via gollek-cli dependencyManagement)
  → gollek-bom dependencyManagement (which imports quarkus-bom)
  → quarkus-bom

Key: Quarkus BOM must win for all Jakarta EE APIs.
```

### Files to keep CDI-version-free

| File | Rule |
|------|------|
| `pom.xml` (parent) | ✅ No Jakarta EE pins |
| `bom/pom.xml` | ✅ No Jakarta EE pins |
| `ui/gollek-cli/pom.xml` | ✅ No explicit `cdi-api` dep |
| Any other module pom | ✅ Never pin CDI/inject/validation |

---

## AccelTensor / FFM Notes

- All tensor allocations use `Arena.ofAuto()` — GC-managed, no explicit `close()` required for arena lifetime.
- `AccelTensor.close()` sets `closed=true` but does **not** close the arena (commented out) — this is intentional for the GraalVM build.
- `AccelOps` lazy-initializes Accelerate framework FFM handles via double-checked locking. These are safe for native-image because `SymbolLookup.libraryLookup(...)` is called at runtime.
- `Arena.global()` is used for the Accelerate `SymbolLookup` — this is intentional and safe.

---

## Build Matrix (Key Constraints)

| Constraint | Value |
|-----------|-------|
| `jdk.incubator.vector` | Required for `AccelOps` SIMD paths |
| `--enable-preview` | Required for Java 25 preview features |
| `-H:-SharedArenaSupport` | Must disable — conflicts with Vector API in GraalVM 25 |
| `-H:+VectorAPISupport` | Enables SIMD in the native image |
| `native-image-xmx` | 12g minimum for this codebase size |
| Max `_Bean` rules | Zero — never add package-level rules for CDI packages |

---

## References
- Conversation: `9007f635-e3c5-4360-b924-d54ff83d4e4c`
- [`native_build_stabilization.md`](file:///Users/bhangun/.gemini/antigravity/brain/9007f635-e3c5-4360-b924-d54ff83d4e4c/artifacts/native_build_stabilization.md)
- [`application.properties`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/ui/gollek-cli/src/main/resources/application.properties)
- [`bom/pom.xml`](file:///Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/bom/pom.xml)
