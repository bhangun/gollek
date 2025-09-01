package llm

import (
	"context"

	"github.com/bhangun/wayang-inference/pkg/types"
)

// Engine defines the interface for LLM inference engines
type Engine interface {
	// Initialize loads the model and prepares the engine
	Initialize() error

	// Generate generates text completion for the given request
	Generate(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error)

	// GenerateStream generates streaming text completion for the given request
	GenerateStream(ctx context.Context, req *types.CompletionRequest) (<-chan StreamToken, error)

	// GetModelInfo returns information about the loaded model
	GetModelInfo() *types.ModelInfo

	// IsHealthy returns true if the engine is ready to serve requests
	IsHealthy() bool

	// GetStats returns engine statistics
	GetStats() *EngineStats

	// Close cleans up resources
	Close() error
}

// StreamToken represents a token in a streaming response
type StreamToken struct {
	Token        string
	IsComplete   bool
	Error        error
	FinishReason string
}

// EngineStats contains engine performance statistics
type EngineStats struct {
	ModelLoaded         bool    `json:"model_loaded"`
	GPULayersLoaded     int     `json:"gpu_layers_loaded"`
	MemoryUsage         uint64  `json:"memory_usage_bytes"`
	TokensProcessed     int64   `json:"tokens_processed"`
	RequestsServed      int64   `json:"requests_served"`
	AverageTokensPerSec float64 `json:"average_tokens_per_sec"`
}
