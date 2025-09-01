package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bhangun/wayang-inference/internal/config"
	"github.com/bhangun/wayang-inference/internal/server"
	"github.com/bhangun/wayang-inference/pkg/logger"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const (
	Version     = "1.0.0"
	ServiceName = "github.com/bhangun/wayang-inference"
)

var (
	configFile  string
	gpuLayers   int
	modelPath   string
	host        string
	port        int
	workers     int
	verbose     bool
	logLevel    string
	contextSize int
)

func main() {
	rootCmd := &cobra.Command{
		Use:   ServiceName,
		Short: "Production-grade LLM inference server using Go and llama.cpp",
		Long: `A high-performance inference server for Large Language Models using:
- Go for server logic and concurrency management
- llama.cpp for fast CPU/GPU inference
- Support for streaming responses via Server-Sent Events
- Configurable GPU acceleration with CUDA and Metal support`,
		Version: Version,
		RunE:    runServer,
	}

	// Global flags
	rootCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "configuration file path")
	rootCmd.PersistentFlags().IntVarP(&gpuLayers, "gpu-layers", "g", 0, "number of layers to offload to GPU (0 for CPU-only)")
	rootCmd.PersistentFlags().StringVarP(&modelPath, "model", "m", "", "path to the model file")
	rootCmd.PersistentFlags().StringVarP(&host, "host", "H", "0.0.0.0", "server host address")
	rootCmd.PersistentFlags().IntVarP(&port, "port", "p", 8080, "server port")
	rootCmd.PersistentFlags().IntVarP(&workers, "workers", "w", 2, "number of worker threads")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "enable verbose logging")
	rootCmd.PersistentFlags().StringVar(&logLevel, "log-level", "info", "log level (debug, info, warn, error)")
	rootCmd.PersistentFlags().IntVar(&contextSize, "context-size", 2048, "context size for the model")

	// Bind flags to viper
	viper.BindPFlag("llm.gpu_layers", rootCmd.PersistentFlags().Lookup("gpu-layers"))
	viper.BindPFlag("llm.model_path", rootCmd.PersistentFlags().Lookup("model"))
	viper.BindPFlag("server.host", rootCmd.PersistentFlags().Lookup("host"))
	viper.BindPFlag("server.port", rootCmd.PersistentFlags().Lookup("port"))
	viper.BindPFlag("llm.worker_pool_size", rootCmd.PersistentFlags().Lookup("workers"))
	viper.BindPFlag("llm.verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("logs.level", rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("llm.context_size", rootCmd.PersistentFlags().Lookup("context-size"))

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runServer(cmd *cobra.Command, args []string) error {
	// Load configuration
	cfg, err := config.Load(configFile)
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	// Override config with command line flags
	if cmd.Flags().Changed("gpu-layers") {
		cfg.LLM.GPULayers = gpuLayers
	}
	if cmd.Flags().Changed("model") {
		cfg.LLM.ModelPath = modelPath
	}
	if cmd.Flags().Changed("host") {
		cfg.Server.Host = host
	}
	if cmd.Flags().Changed("port") {
		cfg.Server.Port = port
	}
	if cmd.Flags().Changed("workers") {
		cfg.LLM.WorkerPoolSize = workers
	}
	if cmd.Flags().Changed("verbose") {
		cfg.LLM.Verbose = verbose
	}
	if cmd.Flags().Changed("log-level") {
		cfg.Logs.Level = logLevel
	}
	if cmd.Flags().Changed("context-size") {
		cfg.LLM.ContextSize = contextSize
	}

	// Validate final configuration
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Initialize logger
	log, err := logger.NewLogger(&cfg.Logs)
	if err != nil {
		return fmt.Errorf("failed to initialize logger: %w", err)
	}

	log.WithField("config", fmt.Sprintf("%+v", cfg)).Info("Starting server with configuration")

	// Create and start server
	srv, err := server.New(cfg, log)
	if err != nil {
		return fmt.Errorf("failed to create server: %w", err)
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server in goroutine
	serverErr := make(chan error, 1)
	go func() {
		log.WithField("address", cfg.GetServerAddress()).Info("Server starting")
		if err := srv.Start(); err != nil {
			serverErr <- err
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		if err != nil {
			log.WithError(err).Error("Server error")
			return err
		}
	case sig := <-sigChan:
		log.WithField("signal", sig).Info("Received shutdown signal")
	}

	// Graceful shutdown
	log.Info("Shutting down server...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := srv.Stop(shutdownCtx); err != nil {
		log.WithError(err).Error("Error during shutdown")
		return err
	}

	log.Info("Server shutdown complete")
	return nil
}
