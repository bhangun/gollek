/**
 * GollekEngine+Extensions.swift
 * High-level Swift extensions with async/await, batching, streaming, caching.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

import Foundation
import Combine

// MARK: - Async/Await API

public extension GollekEngine {

    // MARK: Model loading

    /// Load a `.litertlm` model from the specified bundle (async).
    func loadModel(fromBundle bundle: Bundle = .main, name: String) async throws {
        try await Task.detached(priority: .userInitiated) {
            guard let path = bundle.path(forResource: name, ofType: "litert") else {
                throw GollekStatus.errorModelLoad
            }
            try self.loadModel(from: path)
        }.value
    }

    /// Load a `.litertlm` model from an absolute file URL.
    func loadModel(at url: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            try self.loadModel(from: url.path)
        }.value
    }

    // MARK: Async inference

    /// Run inference on a background executor.
    func inferAsync(input: Data) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            try self.infer(input: input)
        }.value
    }

    // MARK: Batched inference

    /// Run batched inference asynchronously.
    func inferBatchAsync(inputs: [Data]) async throws -> [Data] {
        try await Task.detached(priority: .userInitiated) {
            try self.inferBatch(inputs: inputs)
        }.value
    }

    // MARK: Streaming inference

    /// Run streaming inference with async sequence.
    func streamInference(input: Data, maxTokens: Int32 = 0) -> AsyncThrowingStream<Data, Error> {
        AsyncThrowingStream { continuation in
            Task.detached(priority: .userInitiated) {
                do {
                    let session = try self.startStreaming(input: input, maxTokens: maxTokens)
                    
                    while true {
                        let (data, isDone) = try session.next()
                        if !data.isEmpty {
                            continuation.yield(data)
                        }
                        if isDone {
                            break
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Convenience initializer

public extension GollekEngine {

    /// Create engine with common defaults and immediately load a model.
    convenience init(modelName: String, bundle: Bundle = .main, config: GollekConfig = GollekConfig()) throws {
        try self.init(config: config)
        guard let path = bundle.path(forResource: modelName, ofType: "litert") else {
            throw GollekStatus.errorModelLoad
        }
        try self.loadModel(from: path)
    }
}

// MARK: - Model Cache

/// LRU cache for loaded models with memory quota management.
public actor GollekModelCache {
    private struct CacheEntry {
        let engine: GollekEngine
        let approxBytes: UInt64
        var lastAccessed: Date
        
        init(engine: GollekEngine, approxBytes: UInt64) {
            self.engine = engine
            self.approxBytes = approxBytes
            self.lastAccessed = Date()
        }
    }
    
    private var cache: [String: CacheEntry] = [:]
    private var currentBytes: UInt64 = 0
    private let maxBytes: UInt64
    
    public init(maxBytes: UInt64 = 512 * 1024 * 1024) { // 512 MB default
        self.maxBytes = maxBytes
    }
    
    /// Get or create a cached engine for the given model.
    public func getOrCreate(modelName: String, bundle: Bundle = .main, config: GollekConfig = GollekConfig()) async throws -> GollekEngine {
        let cacheKey = "\(bundle.bundlePath)/\(modelName)"
        
        // Check cache first
        if var entry = cache[cacheKey] {
            entry.lastAccessed = Date()
            cache[cacheKey] = entry
            return entry.engine
        }
        
        // Load new model
        let engine = try GollekEngine(modelName: modelName, bundle: bundle, config: config)
        
        // Estimate model size
        if let modelPath = bundle.path(forResource: modelName, ofType: "litert") {
            let attrs = try? FileManager.default.attributesOfItem(atPath: modelPath)
            let fileSize = attrs?[.size] as? UInt64 ?? 0
            
            // Evict if needed
            while currentBytes + fileSize > maxBytes && !cache.isEmpty {
                evictOldest()
            }
            
            cache[cacheKey] = CacheEntry(engine: engine, approxBytes: fileSize)
            currentBytes += fileSize
        } else {
            cache[cacheKey] = CacheEntry(engine: engine, approxBytes: 0)
        }
        
        return engine
    }
    
    /// Warm up a model by running a dummy inference.
    public func warmUp(modelName: String, bundle: Bundle = .main, config: GollekConfig = GollekConfig(), dummyInput: Data) async throws {
        let engine = try await getOrCreate(modelName: modelName, bundle: bundle, config: config)
        _ = try await engine.inferAsync(input: dummyInput)
    }
    
    /// Evict a specific model from cache.
    public func evict(modelName: String, bundle: Bundle = .main) {
        let cacheKey = "\(bundle.bundlePath)/\(modelName)"
        if let entry = cache.removeValue(forKey: cacheKey) {
            currentBytes -= entry.approxBytes
        }
    }
    
    /// Clear all cached models.
    public func clear() {
        cache.removeAll()
        currentBytes = 0
    }
    
    private func evictOldest() {
        guard let oldestKey = cache.min(by: { $0.value.lastAccessed < $1.value.lastAccessed })?.key else {
            return
        }
        evict(modelName: oldestKey.components(separator: "/").last ?? "")
    }
}

// MARK: - Batching Manager

/// Manages batching of inference requests for improved throughput.
public actor GollekBatchingManager {
    private struct PendingRequest {
        let id: Int
        let input: Data
        let continuation: CheckedContinuation<Data, Error>
    }
    
    private let engine: GollekEngine
    private let maxDelay: Duration
    private let maxBatchSize: Int
    
    private var pending: [PendingRequest] = []
    private var batchTimer: Task<Void, Never>?
    private var nextId: Int = 0
    
    public init(engine: GollekEngine, maxDelay: Duration = .milliseconds(10), maxBatchSize: Int = 32) {
        self.engine = engine
        self.maxDelay = maxDelay
        self.maxBatchSize = maxBatchSize
    }
    
    /// Submit a single input, get a Future that completes with the output.
    public func submit(_ input: Data) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            let request = PendingRequest(id: nextId, input: input, continuation: continuation)
            pending.append(request)
            scheduleBatch()
        }
    }
    
    private func scheduleBatch() {
        guard batchTimer == nil else { return }
        
        batchTimer = Task.detached { [weak self] in
            try? await Task.sleep(for: self?.maxDelay ?? .milliseconds(10))
            await self?.processBatch()
        }
    }
    
    private func processBatch() {
        batchTimer = nil
        
        guard !pending.isEmpty else { return }
        
        // Take up to maxBatchSize requests
        let batch = pending.prefix(maxBatchSize)
        pending.removeFirst(min(batch.count, maxBatchSize))
        
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            
            do {
                let inputs = batch.map { $0.input }
                let outputs = try await self.engine.inferBatchAsync(inputs: inputs)
                
                for (i, request) in batch.enumerated() {
                    if i < outputs.count {
                        request.continuation.resume(returning: outputs[i])
                    } else {
                        request.continuation.resume(throwing: GollekStatus.errorInvoke)
                    }
                }
            } catch {
                for request in batch {
                    request.continuation.resume(throwing: error)
                }
            }
            
            // Schedule next batch if pending requests remain
            if !self.pending.isEmpty {
                await self.scheduleBatch()
            }
        }
    }
}

// MARK: - Combine Publishers

@available(iOS 13.0, macOS 10.15, *)
public extension GollekEngine {
    
    /// Run inference and return a Combine publisher.
    func inferPublisher(input: Data) -> AnyPublisher<Data, Error> {
        Future { promise in
            Task.detached(priority: .userInitiated) {
                do {
                    let result = try self.infer(input: input)
                    promise(.success(result))
                } catch {
                    promise(.failure(error))
                }
            }
        }
        .eraseToAnyPublisher()
    }
}
