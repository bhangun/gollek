# Gollek Core — Module Guide

This directory contains the core building blocks for Gollek. Each module has a focused scope and clear dependency direction.

## Module Structure

### **gollek-spi** (Interfaces & Contracts)
**Purpose**: Public APIs, interfaces, and value objects that other modules depend on

**Key Principle**: Only interfaces, DTOs, and exceptions. No implementations.

---

### **gollek-core** (Core Domain Logic)
**Purpose**: Core business logic, domain models, and base implementations

**Key Principle**: Domain logic, no framework-specific code (Jakarta, Quarkus, etc.)

---

### **gollek-model-repo-core** (Model Repository Layer)
**Purpose**: Model discovery, loading, and repository management


**Key Principle**: Focus on model metadata, discovery, and artifact management. No inference execution logic.

---

### **gollek-provider-core** (Provider SPI)
**Purpose**: Service Provider Interface for pluggable model runners

**Key Principle**: Clean separation between model repository (metadata) and provider (execution)

---

### **gollek-engine** (Inference Engine Implementation)
**Purpose**: Concrete implementations of inference pipeline and orchestration

**Key Principle**: Framework-specific implementations (Jakarta CDI, Quarkus), orchestration logic

---

### **gollek-infrastructure** (Infrastructure & Integration)
**Purpose**: Framework integration, REST resources, persistence

**Key Principle**: All infrastructure concerns - HTTP, persistence, monitoring, plugin loading

---

## Capability Map (Quick)

* **API Contracts**: `inference-gollek/core/gollek-spi/`
* **Domain + Policy**: `inference-gollek/core/gollek-core/`
* **Engine Orchestration**: `inference-gollek/core/gollek-engine/`
* **Model Registry & Artifacts**: `inference-gollek/core/gollek-model-repo-core/`
* **Provider SPI**: `inference-gollek/core/gollek-provider-core/`
* **Infrastructure**: `inference-gollek/core/gollek-infrastructure/`
* **Plugin API**: `inference-gollek/core/gollek-spi/`

## Error Codes

Generate docs for the centralized error codes:

```bash
./scripts/generate-error-codes.sh
```

## Dependency Flow

The modules should depend on each other in this order (no circular dependencies):

```mermaid
graph TD
    A[gollek-spi] 
    C[gollek-core]
    D[gollek-model-repo-core]
    E[gollek-provider-core]
    F[gollek-engine]
    G[gollek-infrastructure]

    %% Tier 1: Base Contracts
    C --> A
    
    %% Tier 2: Domain Logic & SPIs
    D --> C
    E --> C
    E --> B

    %% Tier 3: Core Implementation (Engine)
    F --> D
    F --> E
    F --> C
    F --> B

    %% Tier 4: Infrastructure & Integration
    G --> F
    G --> A
    
    style A fill:#e1f5ff,stroke:#0077c8,stroke-width:2px
    style B fill:#e1f5ff,stroke:#0077c8,stroke-width:2px
    style C fill:#fff4e1,stroke:#d99e00,stroke-width:2px
    style D fill:#fff4e1,stroke:#d99e00,stroke-width:2px
    style E fill:#fff4e1,stroke:#d99e00,stroke-width:2px
    style F fill:#ffe1e1,stroke:#c80000,stroke-width:2px
    style G fill:#f0e1ff,stroke:#6f00c8,stroke-width:2px
```

**Legend**:
- **Blue** (gollek-spi): **Contracts & APIs** - Stable interfaces, minimal dependencies
- **Yellow** (gollek-core, model-repo, provider-core): **Domain Layer** - Business logic & SPI definitions
- **Red** (gollek-engine): **Application Layer** - Orchestration, reliability patterns, implementation
- **Purple** (gollek-infrastructure): **Infrastructure Layer** - Framework integration (Quarkus/REST), adapters

---

---

## Best Practices

### 1. **Single Responsibility Principle**
- Each module should have ONE clear purpose
- If you can't describe a module's purpose in one sentence, it's doing too much

### 2. **Acyclic Dependencies**
- Never allow circular dependencies between modules
- Use interfaces in lower-level modules to break cycles

### 3. **Stable Dependencies Principle**
- Depend on modules that change less frequently
- `gollek-spi` should be the most stable (rarely changes)
- `gollek-infrastructure` can change frequently

### 4. **Interface Segregation**
- Put interfaces in the module that defines the abstraction
- Put implementations in the module that provides the functionality

### 5. **Naming Conventions**
- **Interfaces**: Use descriptive nouns (`ModelRepository`, `InferenceEngine`)
- **Implementations**: Prefix with implementation strategy (`Default`, `Cached`, `Enhanced`)
- **Abstract Classes**: Prefix with `Abstract` (`AbstractPlugin`)
- **DTOs**: Suffix based on purpose (`Request`, `Response`, `Metadata`)

---
