package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bhangun/wayang-inference/internal/config"
	"github.com/bhangun/wayang-inference/pkg/types"

	// llama "github.com/go-skynet/go-llama.cpp"
	"github.com/sirupsen/logrus"
)

// LlamaEngine implements the Engine interface using llama.cpp
type LlamaEngine struct {
	config *config.LLMConfig
	llm    *llama.LLama
	mutex  sync.RWMutex

	// Statistics
	stats *EngineStats

	// Model info
	modelInfo *types.ModelInfo

	logger *logrus.Entry
}

// NewLlamaEngine creates a new LlamaEngine instance
func NewLlamaEngine(cfg *config.LLMConfig, logger *logrus.Logger) *LlamaEngine {
	return &LlamaEngine{
		config: cfg,
		stats: &EngineStats{
			ModelLoaded:     false,
			GPULayersLoaded: 0,
			TokensProcessed: 0,
			RequestsServed:  0,
		},
		logger: logger.WithField("component", "llama-engine"),
	}
}

// Initialize loads the model and prepares the engine
func (e *LlamaEngine) Initialize() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	e.logger.WithFields(logrus.Fields{
		"model_path":   e.config.ModelPath,
		"context_size": e.config.ContextSize,
		"gpu_layers":   e.config.GPULayers,
		"threads":      e.config.Threads,
	}).Info("Initializing Llama engine")

	// Prepare llama.cpp options
	opts := []llama.ModelOption{
		llama.SetContext(e.config.ContextSize),
		llama.SetGPULayers(e.config.GPULayers),
		llama.EnableF16Memory,
	}

	// Add memory mapping if enabled
	if e.config.UseMMap {
		opts = append(opts, llama.EnableMMap)
	}

	// Add memory locking if enabled
	if e.config.UseMLock {
		opts = append(opts, llama.EnableMLock)
	}

	// Set thread count
	if e.config.Threads > 0 {
		opts = append(opts, llama.SetThreads(e.config.Threads))
	}

	// Enable verbose logging if configured
	if e.config.Verbose {
		opts = append(opts, llama.EnableVerbose)
	}

	// Load the model
	llm, err := llama.New(e.config.ModelPath, opts...)
	if err != nil {
		e.logger.WithError(err).Error("Failed to load model")
		return fmt.Errorf("failed to load model: %w", err)
	}

	e.llm = llm
	e.stats.ModelLoaded = true
	e.stats.GPULayersLoaded = e.config.GPULayers

	// Extract model information
	e.modelInfo = &types.ModelInfo{
		Name:         extractModelName(e.config.ModelPath),
		Architecture: "llama",
		ContextSize:  e.config.ContextSize,
	}

	e.logger.Info("Llama engine initialized successfully")
	return nil
}

// Generate generates text completion for the given request
func (e *LlamaEngine) Generate(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	if !e.IsHealthy() {
		return nil, fmt.Errorf("engine is not ready")
	}

	e.mutex.RLock()
	defer e.mutex.RUnlock()

	startTime := time.Now()
	defer func() {
		atomic.AddInt64(&e.stats.RequestsServed, 1)
		e.updateTokensPerSec(time.Since(startTime))
	}()

	// Set default values
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 150
	}

	temperature := req.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}

	topP := req.TopP
	if topP <= 0 {
		topP = 0.9
	}

	topK := req.TopK
	if topK <= 0 {
		topK = 40
	}

	repeatPenalty := req.RepeatPenalty
	if repeatPenalty <= 0 {
		repeatPenalty = 1.1
	}

	// Prepare prediction options
	opts := []llama.PredictOption{
		llama.SetTokens(maxTokens),
		llama.SetTemperature(temperature),
		llama.SetTopP(topP),
		llama.SetTopK(topK),
		llama.SetPenalty(repeatPenalty),
		llama.SetBatch(e.config.BatchSize),
	}

	if req.Seed > 0 {
		opts = append(opts, llama.SetSeed(req.Seed))
	}

	if len(req.Stop) > 0 {
		for _, stop := range req.Stop {
			opts = append(opts, llama.SetStopWords(stop))
		}
	}

	// Generate completion
	result, err := e.llm.Predict(req.Prompt, opts...)
	if err != nil {
		e.logger.WithError(err).Error("Prediction failed")
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// Count tokens (approximate)
	promptTokens := len(strings.Fields(req.Prompt))
	completionTokens := len(strings.Fields(result))
	atomic.AddInt64(&e.stats.TokensProcessed, int64(promptTokens+completionTokens))

	response := &types.CompletionResponse{
		ID:      generateRequestID(),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   e.modelInfo.Name,
		Choices: []types.Choice{
			{
				Index:        0,
				Text:         result,
				FinishReason: "stop", // TODO: Implement proper finish reason detection
			},
		},
		Usage: types.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	return response, nil
}

// GenerateStream generates streaming text completion for the given request
func (e *LlamaEngine) GenerateStream(ctx context.Context, req *types.CompletionRequest) (<-chan StreamToken, error) {
	if !e.IsHealthy() {
		return nil, fmt.Errorf("engine is not ready")
	}

	e.mutex.RLock()
	defer e.mutex.RUnlock()

	tokenCh := make(chan StreamToken, 100)

	go func() {
		defer close(tokenCh)
		defer func() {
			atomic.AddInt64(&e.stats.RequestsServed, 1)
		}()

		// Set default values (same as Generate method)
		maxTokens := req.MaxTokens
		if maxTokens <= 0 {
			maxTokens = 150
		}

		temperature := req.Temperature
		if temperature <= 0 {
			temperature = 0.7
		}

		topP := req.TopP
		if topP <= 0 {
			topP = 0.9
		}

		topK := req.TopK
		if topK <= 0 {
			topK = 40
		}

		repeatPenalty := req.RepeatPenalty
		if repeatPenalty <= 0 {
			repeatPenalty = 1.1
		}

		// Create a token callback function
		tokenCallback := func(token string) bool {
			select {
			case <-ctx.Done():
				return false // Stop generation
			case tokenCh <- StreamToken{Token: token, IsComplete: false}:
				return true // Continue generation
			}
		}

		// Prepare prediction options with token callback
		opts := []llama.PredictOption{
			llama.SetTokens(maxTokens),
			llama.SetTemperature(temperature),
			llama.SetTopP(topP),
			llama.SetTopK(topK),
			llama.SetPenalty(repeatPenalty),
			llama.SetBatch(e.config.BatchSize),
			llama.SetTokenCallback(tokenCallback),
		}

		if req.Seed > 0 {
			opts = append(opts, llama.SetSeed(req.Seed))
		}

		if len(req.Stop) > 0 {
			for _, stop := range req.Stop {
				opts = append(opts, llama.SetStopWords(stop))
			}
		}

		// Generate completion
		_, err := e.llm.Predict(req.Prompt, opts...)

		// Send final token or error
		if err != nil {
			select {
			case <-ctx.Done():
			case tokenCh <- StreamToken{Error: err}:
			}
		} else {
			select {
			case <-ctx.Done():
			case tokenCh <- StreamToken{IsComplete: true, FinishReason: "stop"}:
			}
		}
	}()

	return tokenCh, nil
}

// GetModelInfo returns information about the loaded model
func (e *LlamaEngine) GetModelInfo() *types.ModelInfo {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return e.modelInfo
}

// IsHealthy returns true if the engine is ready to serve requests
func (e *LlamaEngine) IsHealthy() bool {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	return e.llm != nil && e.stats.ModelLoaded
}

// GetStats returns engine statistics
func (e *LlamaEngine) GetStats() *EngineStats {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	statsCopy := *e.stats
	return &statsCopy
}

// Close cleans up resources
func (e *LlamaEngine) Close() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if e.llm != nil {
		e.llm.Free()
		e.llm = nil
		e.stats.ModelLoaded = false
		e.logger.Info("Llama engine closed")
	}

	return nil
}

// Helper functions

func extractModelName(path string) string {
	// Extract model name from path
	parts := strings.Split(path, "/")
	if len(parts) > 0 {
		name := parts[len(parts)-1]
		// Remove extension
		if idx := strings.LastIndex(name, "."); idx > 0 {
			name = name[:idx]
		}
		return name
	}
	return "unknown"
}

func generateRequestID() string {
	return fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
}

func (e *LlamaEngine) updateTokensPerSec(duration time.Duration) {
	// Simple moving average for tokens per second
	// This is a basic implementation - could be improved with proper metrics
	if duration > 0 {
		seconds := duration.Seconds()
		if seconds > 0 {
			// This is a simplified calculation
			e.stats.AverageTokensPerSec = float64(e.stats.TokensProcessed) / seconds
		}
	}
}
