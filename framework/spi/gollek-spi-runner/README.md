# Gollek Model Runner

This module provides the core abstractions and interfaces for gollek model runner in the Gollek Inference Engine.

## Responsibilities
- Define pure interfaces without hardware dependencies.
- Establish the data structures for gollek model runner operations.
- Ensure strict Separation of Concerns (SoC) from orchestrator-level logic.

## Integration
This module is part of the `framework/core` tier and is consumed by `runtime` implementations.
