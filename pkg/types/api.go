package types

import "time"

// CompletionRequest represents a chat completion request
type CompletionRequest struct {
	Prompt        string   `json:"prompt" binding:"required" example:"What is the meaning of life?"`
	MaxTokens     int      `json:"max_tokens,omitempty" example:"150"`
	Temperature   float32  `json:"temperature,omitempty" example:"0.7"`
	TopP          float32  `json:"top_p,omitempty" example:"0.9"`
	TopK          int      `json:"top_k,omitempty" example:"40"`
	RepeatPenalty float32  `json:"repeat_penalty,omitempty" example:"1.1"`
	Seed          int      `json:"seed,omitempty" example:"42"`
	Stream        bool     `json:"stream,omitempty" example:"false"`
	Stop          []string `json:"stop,omitempty" example:"[\"Human:\", \"AI:\"]"`
}

// CompletionResponse represents a chat completion response
type CompletionResponse struct {
	ID      string   `json:"id" example:"cmpl-123456"`
	Object  string   `json:"object" example:"text_completion"`
	Created int64    `json:"created" example:"1677649420"`
	Model   string   `json:"model" example:"llama-2-7b-chat"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	ID      string         `json:"id" example:"cmpl-123456"`
	Object  string         `json:"object" example:"text_completion.chunk"`
	Created int64          `json:"created" example:"1677649420"`
	Model   string         `json:"model" example:"llama-2-7b-chat"`
	Choices []StreamChoice `json:"choices"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int    `json:"index" example:"0"`
	Text         string `json:"text" example:"The meaning of life is..."`
	FinishReason string `json:"finish_reason" example:"stop"`
}

// StreamChoice represents a streaming completion choice
type StreamChoice struct {
	Index        int     `json:"index" example:"0"`
	Text         string  `json:"text" example:"The"`
	FinishReason *string `json:"finish_reason,omitempty" example:"null"`
}

// Usage represents token usage statistics
type Usage struct {
	PromptTokens     int `json:"prompt_tokens" example:"10"`
	CompletionTokens int `json:"completion_tokens" example:"50"`
	TotalTokens      int `json:"total_tokens" example:"60"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error details
type ErrorDetail struct {
	Message string `json:"message" example:"Invalid request"`
	Type    string `json:"type" example:"invalid_request_error"`
	Code    string `json:"code,omitempty" example:"400"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status    string           `json:"status" example:"healthy"`
	Timestamp time.Time        `json:"timestamp" example:"2023-01-01T00:00:00Z"`
	Version   string           `json:"version" example:"1.0.0"`
	Model     HealthModelInfo  `json:"model"`
	System    HealthSystemInfo `json:"system"`
	Stats     HealthStats      `json:"stats"`
}

// HealthModelInfo contains model information
type HealthModelInfo struct {
	Path      string `json:"path" example:"models/llama-2-7b-chat.q4_0.bin"`
	Loaded    bool   `json:"loaded" example:"true"`
	GPULayers int    `json:"gpu_layers" example:"32"`
}

// HealthSystemInfo contains system information
type HealthSystemInfo struct {
	CPUCount    int    `json:"cpu_count" example:"8"`
	GPUCount    int    `json:"gpu_count" example:"1"`
	MemoryUsage string `json:"memory_usage" example:"2.1GB"`
	Platform    string `json:"platform" example:"linux/amd64"`
}

// HealthStats contains runtime statistics
type HealthStats struct {
	ActiveRequests int           `json:"active_requests" example:"3"`
	QueuedRequests int           `json:"queued_requests" example:"1"`
	TotalRequests  int64         `json:"total_requests" example:"1234"`
	AverageLatency time.Duration `json:"average_latency" example:"2.5s"`
	Uptime         time.Duration `json:"uptime" example:"24h30m"`
}

// ModelInfo represents model metadata
type ModelInfo struct {
	Name         string `json:"name" example:"llama-2-7b-chat"`
	Architecture string `json:"architecture" example:"llama"`
	Parameters   string `json:"parameters" example:"7B"`
	Quantization string `json:"quantization" example:"q4_0"`
	ContextSize  int    `json:"context_size" example:"2048"`
}

// Validate validates a CompletionRequest
func (r *CompletionRequest) Validate() error {
	if r.Prompt == "" {
		return NewAPIError("prompt is required", "invalid_request_error", "400")
	}

	if r.MaxTokens < 0 {
		return NewAPIError("max_tokens must be non-negative", "invalid_request_error", "400")
	}

	if r.Temperature < 0 || r.Temperature > 2 {
		return NewAPIError("temperature must be between 0 and 2", "invalid_request_error", "400")
	}

	if r.TopP < 0 || r.TopP > 1 {
		return NewAPIError("top_p must be between 0 and 1", "invalid_request_error", "400")
	}

	return nil
}

// APIError represents a structured API error
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func (e *APIError) Error() string {
	return e.Message
}

// NewAPIError creates a new API error
func NewAPIError(message, errorType, code string) *APIError {
	return &APIError{
		Message: message,
		Type:    errorType,
		Code:    code,
	}
}
