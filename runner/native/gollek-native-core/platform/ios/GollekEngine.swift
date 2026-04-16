/**
 * GollekEngine.swift
 * Pure Swift wrapper for Gollek C API.
 * 
 * Direct C interop with gollek_engine.h - no Objective-C needed.
 * 
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

import Foundation

// MARK: - C API Types

/// Opaque handle to Gollek engine
public typealias GollekEngineHandle = OpaquePointer
/// Opaque handle to streaming session
public typealias GollekStreamSessionHandle = OpaquePointer

// MARK: - Enums

/// Hardware delegate options
@frozen public enum GollekDelegate: Int32 {
    case none = 0
    case cpu = 1
    case gpu = 2
    case nnapi = 3
    case hexagon = 4
    case coreml = 5
    case auto = 6
}

/// Status codes
@frozen public enum GollekStatus: Int32, Error, LocalizedError {
    case ok = 0
    case error = 1
    case errorModelLoad = 2
    case errorAllocTensors = 3
    case errorInvoke = 4
    case errorInvalidArg = 5
    case errorNotInitialized = 6
    case errorDelegateFailed = 7
    
    public var errorDescription: String? {
        switch self {
        case .ok: return "OK"
        case .error: return "Generic error"
        case .errorModelLoad: return "Model load failed"
        case .errorAllocTensors: return "Tensor allocation failed"
        case .errorInvoke: return "Interpreter invoke failed"
        case .errorInvalidArg: return "Invalid argument"
        case .errorNotInitialized: return "Engine not initialized"
        case .errorDelegateFailed: return "Delegate creation failed"
        }
    }
}

/// Tensor data types
@frozen public enum GollekTensorType: Int32 {
    case float32 = 0
    case float16 = 1
    case int32 = 2
    case uint8 = 3
    case int64 = 4
    case int8 = 6
    case bool = 9
}

// MARK: - Configuration

/// Engine configuration
public struct GollekConfig {
    public var numThreads: Int32
    public var delegate: GollekDelegate
    public var enableXnnpack: Bool
    public var useMemoryPool: Bool
    public var poolSizeBytes: Int
    
    public init(
        numThreads: Int32 = 4,
        delegate: GollekDelegate = .auto,
        enableXnnpack: Bool = true,
        useMemoryPool: Bool = true,
        poolSizeBytes: Int = 0
    ) {
        self.numThreads = numThreads
        self.delegate = delegate
        self.enableXnnpack = enableXnnpack
        self.useMemoryPool = useMemoryPool
        self.poolSizeBytes = poolSizeBytes
    }
}

// MARK: - Tensor Info

/// Tensor descriptor
public struct GollekTensorInfo {
    public let name: String
    public let type: GollekTensorType
    public let dims: [Int32]
    public let byteSize: Int
    public let scale: Float
    public let zeroPoint: Int32
    
    init(fromC info: gollek_tensor_info_c) {
        self.name = info.name.map { String(cString: $0) } ?? ""
        self.type = GollekTensorType(rawValue: info.type) ?? .float32
        self.dims = Array(info.dims.prefix(Int(info.num_dims)))
        self.byteSize = info.byte_size
        self.scale = info.scale
        self.zeroPoint = info.zero_point
    }
}

// C struct for tensor info (matches gollek_engine.h)
private struct gollek_tensor_info_c {
    var name: UnsafePointer<CChar>?
    var type: Int32
    var dims: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
    var num_dims: Int32
    var byte_size: Int
    var scale: Float
    var zero_point: Int32
}

// MARK: - Metrics

/// Performance metrics
public struct GollekMetrics {
    public let totalInferences: UInt64
    public let failedInferences: UInt64
    public let avgLatencyMs: Double
    public let p50LatencyMs: Double
    public let p95LatencyMs: Double
    public let p99LatencyMs: Double
    public let peakMemoryBytes: UInt64
    public let currentMemoryBytes: UInt64
    public let activeDelegate: GollekDelegate
    
    init(fromC metrics: gollek_metrics_c) {
        self.totalInferences = metrics.total_inferences
        self.failedInferences = metrics.failed_inferences
        self.avgLatencyMs = Double(metrics.avg_latency_us) / 1000.0
        self.p50LatencyMs = Double(metrics.p50_latency_us) / 1000.0
        self.p95LatencyMs = Double(metrics.p95_latency_us) / 1000.0
        self.p99LatencyMs = Double(metrics.p99_latency_us) / 1000.0
        self.peakMemoryBytes = metrics.peak_memory_bytes
        self.currentMemoryBytes = metrics.current_memory_bytes
        self.activeDelegate = GollekDelegate(rawValue: Int32(metrics.active_delegate)) ?? .cpu
    }
}

// C struct for metrics (matches gollek_engine.h)
private struct gollek_metrics_c {
    var total_inferences: UInt64
    var failed_inferences: UInt64
    var total_latency_us: UInt64
    var avg_latency_us: UInt64
    var p50_latency_us: UInt64
    var p95_latency_us: UInt64
    var p99_latency_us: UInt64
    var peak_memory_bytes: UInt64
    var current_memory_bytes: UInt64
    var active_delegate: Int32
}

// MARK: - C API Imports

// These are imported from gollek_engine.h via modulemap or bridging header
@_silgen_name("gollek_engine_create")
private func gollek_engine_create(_ config: UnsafePointer<GollekConfigC>?) -> GollekEngineHandle?

@_silgen_name("gollek_engine_destroy")
private func gollek_engine_destroy(_ engine: GollekEngineHandle?)

@_silgen_name("gollek_load_model_from_file")
private func gollek_load_model_from_file(_ engine: GollekEngineHandle?, _ path: UnsafePointer<CChar>?) -> Int32

@_silgen_name("gollek_load_model_from_buffer")
private func gollek_load_model_from_buffer(_ engine: GollekEngineHandle?, _ data: UnsafeRawPointer?, _ size: Int) -> Int32

@_silgen_name("gollek_get_input_count")
private func gollek_get_input_count(_ engine: GollekEngineHandle?) -> Int32

@_silgen_name("gollek_get_output_count")
private func gollek_get_output_count(_ engine: GollekEngineHandle?) -> Int32

@_silgen_name("gollek_get_input_info")
private func gollek_get_input_info(_ engine: GollekEngineHandle?, _ index: Int32, _ out: UnsafeMutablePointer<gollek_tensor_info_c>?) -> Int32

@_silgen_name("gollek_get_output_info")
private func gollek_get_output_info(_ engine: GollekEngineHandle?, _ index: Int32, _ out: UnsafeMutablePointer<gollek_tensor_info_c>?) -> Int32

@_silgen_name("gollek_set_input")
private func gollek_set_input(_ engine: GollekEngineHandle?, _ index: Int32, _ src: UnsafeRawPointer?, _ bytes: Int) -> Int32

@_silgen_name("gollek_invoke")
private func gollek_invoke(_ engine: GollekEngineHandle?) -> Int32

@_silgen_name("gollek_get_output")
private func gollek_get_output(_ engine: GollekEngineHandle?, _ index: Int32, _ dst: UnsafeMutableRawPointer?, _ dstBytes: Int) -> Int32

@_silgen_name("gollek_infer")
private func gollek_infer(_ engine: GollekEngineHandle?, _ input: UnsafeRawPointer?, _ inputBytes: Int, _ output: UnsafeMutableRawPointer?, _ outputBytes: Int) -> Int32

@_silgen_name("gollek_set_batch_input")
private func gollek_set_batch_input(_ engine: GollekEngineHandle?, _ index: Int32, _ inputs: UnsafePointer<UnsafeRawPointer?>?, _ inputBytes: UnsafePointer<Int>?, _ numInputs: Int32) -> Int32

@_silgen_name("gollek_get_batch_output")
private func gollek_get_batch_output(_ engine: GollekEngineHandle?, _ index: Int32, _ outputs: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ outputBytes: UnsafePointer<Int>?, _ numOutputs: Int32) -> Int32

@_silgen_name("gollek_start_streaming")
private func gollek_start_streaming(_ engine: GollekEngineHandle?, _ input: UnsafeRawPointer?, _ inputBytes: Int, _ maxTokens: Int32, _ outSession: UnsafeMutablePointer<GollekStreamSessionHandle?>?) -> Int32

@_silgen_name("gollek_stream_next")
private func gollek_stream_next(_ session: GollekStreamSessionHandle?, _ output: UnsafeMutableRawPointer?, _ outputBytes: Int, _ actualBytes: UnsafeMutablePointer<Int>?, _ isDone: UnsafeMutablePointer<Int32>?) -> Int32

@_silgen_name("gollek_end_streaming")
private func gollek_end_streaming(_ session: GollekStreamSessionHandle?)

@_silgen_name("gollek_get_metrics")
private func gollek_get_metrics(_ engine: GollekEngineHandle?, _ out: UnsafeMutablePointer<gollek_metrics_c>?) -> Int32

@_silgen_name("gollek_reset_metrics")
private func gollek_reset_metrics(_ engine: GollekEngineHandle?) -> Int32

@_silgen_name("gollek_last_error")
private func gollek_last_error(_ engine: GollekEngineHandle?) -> UnsafePointer<CChar>?

@_silgen_name("gollek_version")
private func gollek_version() -> UnsafePointer<CChar>?

// C-compatible config struct
private struct GollekConfigC {
    var num_threads: Int32
    var delegate: Int32
    var enable_xnnpack: Int32
    var use_memory_pool: Int32
    var pool_size_bytes: Int
}

// MARK: - Streaming Session

/// Streaming inference session wrapper
public class GollekStreamSession {
    private let handle: GollekStreamSessionHandle
    private var isDone = false
    
    init(handle: GollekStreamSessionHandle) {
        self.handle = handle
    }
    
    deinit {
        gollek_end_streaming(handle)
    }
    
    /// Get next chunk from streaming session
    public func next() throws -> (data: Data, isDone: Bool) {
        if isDone {
            return (data: Data(), isDone: true)
        }
        
        // Allocate buffer for output
        let bufferSize = 4096 // Adjust based on your model
        let outputBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { outputBuffer.deallocate() }
        
        var actualBytes: Int = 0
        var done: Int32 = 0
        
        let status = gollek_stream_next(handle, outputBuffer, bufferSize, &actualBytes, &done)
        
        guard status == 0 else {
            throw GollekStatus(rawValue: status) ?? GollekStatus.error
        }
        
        isDone = done != 0
        let data = Data(bytes: outputBuffer, count: actualBytes)
        
        return (data: data, isDone: isDone)
    }
}

// MARK: - Engine

/// Gollek LiteRT Inference Engine - Pure Swift API
public final class GollekEngine {
    private let handle: GollekEngineHandle
    private var isDestroyed = false
    
    /// Create a new engine instance
    public init(config: GollekConfig = GollekConfig()) throws {
        let cConfig = GollekConfigC(
            num_threads: config.numThreads,
            delegate: config.delegate.rawValue,
            enable_xnnpack: config.enableXnnpack ? 1 : 0,
            use_memory_pool: config.useMemoryPool ? 1 : 0,
            pool_size_bytes: config.poolSizeBytes
        )
        
        guard let handle = withUnsafePointer(to: cConfig, { gollek_engine_create($0) }) else {
            throw GollekStatus.errorNotInitialized
        }
        
        self.handle = handle
    }
    
    deinit {
        destroy()
    }
    
    /// Destroy engine and free resources
    public func destroy() {
        guard !isDestroyed else { return }
        isDestroyed = true
        gollek_engine_destroy(handle)
    }
    
    /// Check if engine is valid
    public var isValid: Bool {
        return !isDestroyed
    }
    
    // MARK: - Model Loading
    
    /// Load model from file path
    public func loadModel(from path: String) throws {
        try checkValid()
        let status = path.withCString { gollek_load_model_from_file(handle, $0) }
        try checkStatus(status)
    }
    
    /// Load model from Data buffer
    public func loadModel(from data: Data) throws {
        try checkValid()
        let status = data.withUnsafeBytes { ptr in
            gollek_load_model_from_buffer(handle, ptr.baseAddress, data.count)
        }
        try checkStatus(status)
    }
    
    // MARK: - Tensor Info
    
    /// Get input tensor count
    public var inputCount: Int {
        return Int(gollek_get_input_count(handle))
    }
    
    /// Get output tensor count
    public var outputCount: Int {
        return Int(gollek_get_output_count(handle))
    }
    
    /// Get input tensor info at index
    public func inputInfo(at index: Int) throws -> GollekTensorInfo {
        try checkValid()
        var info = gollek_tensor_info_c(
            name: nil, type: 0,
            dims: (0,0,0,0,0,0,0,0), num_dims: 0,
            byte_size: 0, scale: 0, zero_point: 0
        )
        let status = gollek_get_input_info(handle, Int32(index), &info)
        try checkStatus(status)
        return GollekTensorInfo(fromC: info)
    }
    
    /// Get output tensor info at index
    public func outputInfo(at index: Int) throws -> GollekTensorInfo {
        try checkValid()
        var info = gollek_tensor_info_c(
            name: nil, type: 0,
            dims: (0,0,0,0,0,0,0,0), num_dims: 0,
            byte_size: 0, scale: 0, zero_point: 0
        )
        let status = gollek_get_output_info(handle, Int32(index), &info)
        try checkStatus(status)
        return GollekTensorInfo(fromC: info)
    }
    
    // MARK: - Inference
    
    /// Set input tensor at index
    public func setInput(_ data: Data, at index: Int = 0) throws {
        try checkValid()
        let status = data.withUnsafeBytes { ptr in
            gollek_set_input(handle, Int32(index), ptr.baseAddress, data.count)
        }
        try checkStatus(status)
    }
    
    /// Run inference
    public func invoke() throws {
        try checkValid()
        let status = gollek_invoke(handle)
        try checkStatus(status)
    }
    
    /// Get output tensor at index
    public func output(at index: Int = 0) throws -> Data {
        try checkValid()
        let info = try outputInfo(at: index)
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: info.byteSize)
        defer { buffer.deallocate() }
        
        let status = gollek_get_output(handle, Int32(index), buffer, info.byteSize)
        try checkStatus(status)
        
        return Data(bytes: buffer, count: info.byteSize)
    }
    
    /// Single-call inference (set input, invoke, get output)
    public func infer(input: Data) throws -> Data {
        try checkValid()
        let outputInfo = try self.outputInfo(at: 0)
        let outputBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: outputInfo.byteSize)
        defer { outputBuffer.deallocate() }
        
        let status = input.withUnsafeBytes { inputPtr in
            gollek_infer(handle, inputPtr.baseAddress, input.count, outputBuffer, outputInfo.byteSize)
        }
        try checkStatus(status)
        
        return Data(bytes: outputBuffer, count: outputInfo.byteSize)
    }
    
    // MARK: - Batched Inference
    
    /// Set batched input tensors
    public func setBatchInput(_ inputs: [Data], at index: Int = 0) throws {
        try checkValid()
        guard !inputs.isEmpty else {
            throw GollekStatus.errorInvalidArg
        }
        
        var inputPtrs: [UnsafeRawPointer?] = []
        var inputBytes: [Int] = []
        
        for input in inputs {
            input.withUnsafeBytes { ptr in
                inputPtrs.append(ptr.baseAddress)
                inputBytes.append(input.count)
            }
        }
        
        let status = inputPtrs.withUnsafeBufferPointer { ptrs in
            inputBytes.withUnsafeBufferPointer { bytes in
                gollek_set_batch_input(handle, Int32(index), ptrs.baseAddress, bytes.baseAddress, Int32(inputs.count))
            }
        }
        try checkStatus(status)
    }
    
    /// Get batched output tensors
    public func getBatchOutput(at index: Int = 0, numOutputs: Int) throws -> [Data] {
        try checkValid()
        let info = try outputInfo(at: index)
        let elementBytes = info.byteSize / Int(info.dims[0])
        
        var outputs: [Data] = []
        var outputPtrs: [UnsafeMutableRawPointer?] = []
        var outputBytes: [Int] = []
        
        for _ in 0..<numOutputs {
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: elementBytes)
            outputPtrs.append(buffer)
            outputBytes.append(elementBytes)
        }
        
        defer {
            for ptr in outputPtrs {
                ptr?.deallocate()
            }
        }
        
        let status = outputPtrs.withUnsafeMutableBufferPointer { ptrs in
            outputBytes.withUnsafeBufferPointer { bytes in
                gollek_get_batch_output(handle, Int32(index), ptrs.baseAddress, bytes.baseAddress, Int32(numOutputs))
            }
        }
        try checkStatus(status)
        
        for i in 0..<numOutputs {
            if let ptr = outputPtrs[i] {
                outputs.append(Data(bytes: ptr, count: elementBytes))
            }
        }
        
        return outputs
    }
    
    /// Run batched inference
    public func inferBatch(inputs: [Data]) throws -> [Data] {
        try checkValid()
        try setBatchInput(inputs, at: 0)
        try invoke()
        return try getBatchOutput(at: 0, numOutputs: inputs.count)
    }
    
    // MARK: - Streaming
    
    /// Start streaming inference
    public func startStreaming(input: Data, maxTokens: Int32 = 0) throws -> GollekStreamSession {
        try checkValid()
        var sessionHandle: GollekStreamSessionHandle? = nil
        
        let status = input.withUnsafeBytes { ptr in
            gollek_start_streaming(handle, ptr.baseAddress, input.count, maxTokens, &sessionHandle)
        }
        try checkStatus(status)
        
        guard let sessionHandle = sessionHandle else {
            throw GollekStatus.errorInvoke
        }
        
        return GollekStreamSession(handle: sessionHandle)
    }
    
    // MARK: - Metrics
    
    /// Get performance metrics
    public var metrics: GollekMetrics? {
        var cMetrics = gollek_metrics_c(
            total_inferences: 0, failed_inferences: 0,
            total_latency_us: 0, avg_latency_us: 0,
            p50_latency_us: 0, p95_latency_us: 0, p99_latency_us: 0,
            peak_memory_bytes: 0, current_memory_bytes: 0,
            active_delegate: 0
        )
        
        let status = gollek_get_metrics(handle, &cMetrics)
        guard status == 0 else { return nil }
        
        return GollekMetrics(fromC: cMetrics)
    }
    
    /// Reset performance metrics
    public func resetMetrics() throws {
        try checkValid()
        let status = gollek_reset_metrics(handle)
        try checkStatus(status)
    }
    
    // MARK: - Diagnostics
    
    /// Get last error message
    public var lastError: String {
        guard let cStr = gollek_last_error(handle) else { return "Unknown error" }
        return String(cString: cStr)
    }
    
    /// Get engine version
    public static var version: String {
        guard let cStr = gollek_version() else { return "unknown" }
        return String(cString: cStr)
    }
    
    // MARK: - Private
    
    private func checkValid() throws {
        guard !isDestroyed else {
            throw GollekStatus.errorNotInitialized
        }
    }
    
    private func checkStatus(_ status: Int32) throws {
        guard status != 0 else { return }
        throw GollekStatus(rawValue: status) ?? GollekStatus.error
    }
}
