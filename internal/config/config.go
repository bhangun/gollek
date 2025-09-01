package config

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/viper"
)

// Config holds all configuration for the server
type Config struct {
	Server ServerConfig `yaml:"server" mapstructure:"server"`
	LLM    LLMConfig    `yaml:"llm" mapstructure:"llm"`
	Logs   LogConfig    `yaml:"logs" mapstructure:"logs"`
}

// ServerConfig contains HTTP server configuration
type ServerConfig struct {
	Host           string        `yaml:"host" mapstructure:"host"`
	Port           int           `yaml:"port" mapstructure:"port"`
	ReadTimeout    time.Duration `yaml:"read_timeout" mapstructure:"read_timeout"`
	WriteTimeout   time.Duration `yaml:"write_timeout" mapstructure:"write_timeout"`
	MaxRequestSize int64         `yaml:"max_request_size" mapstructure:"max_request_size"`
}

// LLMConfig contains LLM engine configuration
type LLMConfig struct {
	ModelPath      string        `yaml:"model_path" mapstructure:"model_path"`
	ContextSize    int           `yaml:"context_size" mapstructure:"context_size"`
	GPULayers      int           `yaml:"gpu_layers" mapstructure:"gpu_layers"`
	Threads        int           `yaml:"threads" mapstructure:"threads"`
	BatchSize      int           `yaml:"batch_size" mapstructure:"batch_size"`
	WorkerPoolSize int           `yaml:"worker_pool_size" mapstructure:"worker_pool_size"`
	MaxQueueSize   int           `yaml:"max_queue_size" mapstructure:"max_queue_size"`
	RequestTimeout time.Duration `yaml:"request_timeout" mapstructure:"request_timeout"`
	UseMMap        bool          `yaml:"use_mmap" mapstructure:"use_mmap"`
	UseMLock       bool          `yaml:"use_mlock" mapstructure:"use_mlock"`
	UseFP16        bool          `yaml:"use_fp16" mapstructure:"use_fp16"`
	Verbose        bool          `yaml:"verbose" mapstructure:"verbose"`
}

// LogConfig contains logging configuration
type LogConfig struct {
	Level  string `yaml:"level" mapstructure:"level"`
	Format string `yaml:"format" mapstructure:"format"`
	File   string `yaml:"file" mapstructure:"file"`
}

// DefaultConfig returns a configuration with sensible defaults
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:           "0.0.0.0",
			Port:           8080,
			ReadTimeout:    30 * time.Second,
			WriteTimeout:   300 * time.Second, // Longer timeout for streaming
			MaxRequestSize: 10 * 1024 * 1024,  // 10MB
		},
		LLM: LLMConfig{
			ModelPath:      "models/llama-2-7b-chat.q4_0.bin",
			ContextSize:    2048,
			GPULayers:      0, // CPU-only by default
			Threads:        4,
			BatchSize:      512,
			WorkerPoolSize: 2,
			MaxQueueSize:   100,
			RequestTimeout: 5 * time.Minute,
			UseMMap:        true,
			UseMLock:       false,
			UseFP16:        true,
			Verbose:        false,
		},
		Logs: LogConfig{
			Level:  "info",
			Format: "json",
			File:   "",
		},
	}
}

// Load loads configuration from file, environment variables, and command line flags
func Load(configPath string) (*Config, error) {
	cfg := DefaultConfig()

	// Set up Viper
	viper.SetConfigType("yaml")
	viper.SetEnvPrefix("LLM_SERVER")
	viper.AutomaticEnv()

	// Load from config file if provided
	if configPath != "" {
		viper.SetConfigFile(configPath)
		if err := viper.ReadInConfig(); err != nil {
			if !os.IsNotExist(err) {
				return nil, fmt.Errorf("failed to read config file: %w", err)
			}
		}
	}

	// Unmarshal into config struct
	if err := viper.Unmarshal(cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return cfg, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	// Check if model file exists
	if _, err := os.Stat(c.LLM.ModelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file does not exist: %s", c.LLM.ModelPath)
	}

	// Validate worker pool size
	if c.LLM.WorkerPoolSize <= 0 {
		return fmt.Errorf("worker pool size must be positive, got: %d", c.LLM.WorkerPoolSize)
	}

	// Validate context size
	if c.LLM.ContextSize <= 0 {
		return fmt.Errorf("context size must be positive, got: %d", c.LLM.ContextSize)
	}

	// Validate GPU layers (can be 0 for CPU-only)
	if c.LLM.GPULayers < 0 {
		return fmt.Errorf("GPU layers must be non-negative, got: %d", c.LLM.GPULayers)
	}

	// Validate port
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}

	return nil
}

// GetServerAddress returns the server address in host:port format
func (c *Config) GetServerAddress() string {
	return fmt.Sprintf("%s:%d", c.Server.Host, c.Server.Port)
}
