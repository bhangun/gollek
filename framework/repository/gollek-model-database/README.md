# Gollek Model Database

This module defines the core repository and database interfaces for model weight management in the Gollek Inference Engine.

## Responsibilities
- Provide generic repository abstractions.
- Manage local database schemas for cached models.

## Integration
This module is part of the `framework/repository` tier and is implemented by specific backend modules (e.g. Kaggle, HuggingFace) in `runtime/repository`.
