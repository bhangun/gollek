# Gollek Quota Plugin

The Gollek Quota Plugin provides tenant-level quota enforcement for inference requests in the Gollek platform.

## Overview

This plugin enforces configurable quotas per tenant to prevent abuse and ensure fair resource distribution across tenants. The plugin operates in two phases:

1. **Authorization Phase**: Checks if the tenant has available quota before processing the request
2. **Cleanup Phase**: Releases the reserved quota after the request is processed

## Features

- **Tenant-level quotas**: Each tenant has independent quota limits
- **Configurable limits**: Quotas can be customized per tenant
- **Automatic reservation/release**: Quotas are automatically reserved and released
- **Time-based windows**: Supports configurable time windows (hourly, daily, etc.)

## Architecture

```
┌─────────────────────────────────────┐
│   QuotaEnforcementPlugin            │
│                                     │
│   • Runs in AUTHORIZE phase         │
│   • Checks tenant quota             │
│   • Reserves quota for request      │
└─────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│   QuotaCleanupPlugin                │
│                                     │
│   • Runs in CLEANUP phase           │
│   • Releases reserved quota         │
└─────────────────────────────────────┘
```

## Components

### QuotaEnforcementPlugin
- Runs during the `AUTHORIZE` phase
- Checks if the requesting tenant has available quota
- Reserves quota for the current request
- Throws exception if quota is exceeded

### QuotaCleanupPlugin  
- Runs during the `CLEANUP` phase
- Releases the quota that was reserved for the request
- Ensures quotas are properly freed even if requests fail

### TenantQuotaService
Interface for managing tenant quotas with methods for:
- Checking current quota status
- Reserving quota for requests
- Releasing quota after requests complete
- Getting quota configuration

### DefaultTenantQuotaService
Default in-memory implementation of TenantQuotaService that:
- Stores quota usage per tenant in memory
- Supports configurable limits and time windows
- Automatically resets counters based on time windows

## Configuration

The plugin uses sensible defaults but can be configured per tenant:

- **Default limit**: 1000 requests per hour
- **Time window**: 1 hour (3600000 ms)
- **Units**: Requests
- **Enabled by default**: Yes

## Integration

The plugin integrates with the Gollek plugin system and follows these conventions:

- Plugin ID: `tech.kayys.gollek.policy.quota`
- Execution order: 10 (early in authorization phase)
- Phase: AUTHORIZE
- Lifecycle: Follows standard plugin lifecycle (initialize/shutdown)

## Usage

The plugin works automatically when registered in the Gollek inference engine. No additional configuration is required for basic usage.

For custom quota configurations per tenant, implement a custom `TenantQuotaService` and register it with the DI container.