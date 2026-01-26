# Golek Model Router Plugin

The Golek Model Router Plugin provides intelligent model-to-provider routing capabilities for inference requests in the Golek platform.

## Overview

This plugin determines the optimal provider for a given model based on various factors including performance, cost, latency, and reliability. It makes routing decisions during the ROUTE phase of the inference pipeline.

## Features

- **Intelligent Routing**: Selects the best provider based on multiple scoring factors
- **Multi-Factor Scoring**: Considers performance, cost, latency, and reliability
- **Tenant-Aware**: Makes routing decisions considering tenant-specific requirements
- **Context-Aware**: Adapts routing based on request context (priority, size, etc.)
- **Fallback Mechanisms**: Provides fallback routing when primary options are unavailable
- **Detailed Auditing**: Comprehensive logging of routing decisions for observability

## Architecture

```
┌─────────────────────────────────────┐
│   ModelRouterPlugin                 │
│                                    │
│   • Runs in ROUTE phase           │
│   • Extracts model and context    │
│   • Selects optimal provider      │
│   • Stores routing decision       │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   ModelRouterService Interface      │
│                                    │
│   • Defines routing contract       │
│   • Supports context-aware routing │
│   • Provides scoring mechanisms    │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   DefaultModelRouterService         │
│                                    │
│   • Multi-factor scoring algorithm │
│   • Weighted decision making      │
│   • Fallback mechanisms           │
└─────────────────────────────────────┘
```

## Components

### ModelRouterPlugin
- Runs during the `ROUTE` phase
- Extracts model ID and request context from execution context
- Calls the ModelRouterService to make routing decisions
- Stores routing decisions in context for downstream phases
- Adds audit information for observability

### ModelRouterService
Interface defining model routing capabilities with methods for:
- Basic provider selection
- Context-aware provider selection
- Available provider queries
- Scoring mechanisms
- Detailed routing decisions

### DefaultModelRouterService
Default implementation using:
- Multi-factor scoring algorithm
- Weighted decision making (performance, cost, latency, reliability)
- Tenant-aware routing
- Context-sensitive adjustments
- Fallback mechanisms

### ProviderCapabilities
Represents provider characteristics including:
- Reliability percentage
- Average latency
- Cost per thousand units
- Performance tier

### RoutingDecision
Encapsulates routing outcomes with:
- Selected provider ID
- Scoring information
- Candidate list
- Metadata and timestamps

## Scoring Factors

The router considers multiple factors when selecting a provider:

- **Performance (30%)**: Provider performance metrics
- **Cost (25%)**: Cost-effectiveness of the provider
- **Latency (25%)**: Expected response time
- **Reliability (20%)**: Historical uptime and stability

Weights can be adjusted based on request context (e.g., priority for performance vs cost).

## Fallback Mechanisms

The plugin implements multiple fallback strategies:
- Model-specific fallbacks (e.g., "gpt-*" → OpenAI provider)
- Generic fallback to first available provider
- Graceful degradation when no providers are available

## Integration

The plugin integrates with the Golek plugin system and follows these conventions:

- Plugin ID: `tech.kayys.golek.routing.model`
- Execution order: 1 (early in route phase)
- Phase: ROUTE
- Lifecycle: Follows standard plugin lifecycle (initialize/shutdown)

## Usage

The plugin works automatically when registered in the Golek inference engine. No additional configuration is required for basic usage.

For custom routing policies, implement a custom `ModelRouterService` and register it with the DI container.