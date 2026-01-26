# Golek Content Safety Plugin

The Golek Content Safety Plugin provides content moderation capabilities for inference requests in the Golek platform.

## Overview

This plugin analyzes input content to detect potentially unsafe or inappropriate material before processing inference requests. It uses configurable policies to identify content that violates safety guidelines.

## Features

- **Content Moderation**: Analyzes text content for safety violations
- **Configurable Categories**: Supports multiple safety categories (hate speech, violence, etc.)
- **Confidence Scoring**: Provides confidence levels for detected violations
- **Flexible Policies**: Configurable thresholds and blocking behavior
- **Detailed Logging**: Comprehensive audit trails for compliance

## Architecture

```
┌─────────────────────────────────────┐
│   ContentSafetyPlugin              │
│                                    │
│   • Runs in VALIDATION phase      │
│   • Extracts input content        │
│   • Moderates content             │
│   • Blocks unsafe content         │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   ContentModerator Interface        │
│                                    │
│   • Defines moderation contract    │
│   • Supports category-based checks │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   DefaultContentModerator           │
│                                    │
│   • Keyword-based detection       │
│   • Regex pattern matching        │
│   • Configurable policies         │
└─────────────────────────────────────┘
```

## Components

### ContentSafetyPlugin
- Runs during the `VALIDATION` phase
- Extracts content from various input sources
- Calls the ContentModerator to analyze content
- Blocks requests with unsafe content
- Adds audit information to the execution context

### ContentModerator
Interface defining content moderation capabilities with methods for:
- Basic content moderation
- Category-specific moderation
- Violation detection

### DefaultContentModerator
Default implementation using:
- Keyword-based detection
- Regular expression patterns
- Configurable safety categories
- Confidence scoring

### ContentSafetyConfig
Configuration class supporting:
- Enable/disable safety checks
- Configure safety categories
- Set minimum confidence thresholds
- Control blocking behavior

## Supported Categories

- Hate Speech
- Violence
- Sexual Content
- Self-Harm
- Harassment
- Dangerous Content
- Misinformation

## Configuration

The plugin uses sensible defaults but can be configured:

- **Enabled by default**: Yes
- **Default categories**: hate_speech, violence, self_harm, sexual_content
- **Minimum confidence**: 0.7
- **Block on violation**: Yes
- **Log violations**: Yes

## Integration

The plugin integrates with the Golek plugin system and follows these conventions:

- Plugin ID: `tech.kayys.golek.safety.content`
- Execution order: 20 (during validation phase)
- Phase: VALIDATION
- Lifecycle: Follows standard plugin lifecycle (initialize/shutdown)

## Usage

The plugin works automatically when registered in the Golek inference engine. No additional configuration is required for basic usage.

For custom safety policies, implement a custom `ContentModerator` and register it with the DI container.