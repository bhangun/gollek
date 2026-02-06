# Anthropic Provider for Golek

The Anthropic provider enables integration with Anthropic's Claude models through the Golek inference platform.

## Features

- Support for all Claude models (Claude 3 Opus, Sonnet, Haiku, Claude 2.x)
- Streaming and non-streaming inference
- System message handling
- Configurable parameters (temperature, max tokens, etc.)
- Error handling and retry mechanisms
- Health checking

## Configuration

The provider can be configured with the following parameters:

- `apiKey` (required): Your Anthropic API key
- `baseUrl` (optional): Base URL for the Anthropic API (defaults to `https://api.anthropic.com`)
- `apiVersion` (optional): API version to use (defaults to `2023-06-01`)
- `defaultMaxTokens` (optional): Default max tokens to use if not specified in request
- `defaultTemperature` (optional): Default temperature to use if not specified in request

## Supported Models

- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-2.1`
- `claude-2.0`

## Usage

The provider implements the standard Golek provider interface and can be used like any other provider in the Golek ecosystem.

## Error Handling

The provider includes comprehensive error handling for:
- Invalid API keys
- Network connectivity issues
- API rate limits
- Invalid requests
- Model unavailability
