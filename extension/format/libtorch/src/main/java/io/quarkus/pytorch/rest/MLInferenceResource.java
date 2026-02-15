package io.quarkus.pytorch.rest;

import io.quarkus.pytorch.core.Tensor;
import io.quarkus.pytorch.examples.MNISTNet;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * REST API for PyTorch model inference.
 * Demonstrates real-world Quarkus integration with PyTorch Java bindings.
 */
@Path("/api/ml")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class MLInferenceResource {
    
    @ConfigProperty(name = "pytorch.device", defaultValue = "cpu")
    String device;
    
    @ConfigProperty(name = "pytorch.enable-cuda", defaultValue = "true")
    boolean enableCuda;
    
    private volatile MNISTNet model;
    
    /**
     * Health check endpoint.
     */
    @GET
    @Path("/health")
    public Response health() {
        Map<String, Object> status = new HashMap<>();
        status.put("status", "UP");
        status.put("cuda_available", Tensor.cudaIsAvailable());
        status.put("configured_device", device);
        
        return Response.ok(status).build();
    }
    
    /**
     * Get system information.
     */
    @GET
    @Path("/info")
    public Response systemInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("jvm_version", System.getProperty("java.version"));
        info.put("os", System.getProperty("os.name"));
        info.put("arch", System.getProperty("os.arch"));
        info.put("cuda_available", Tensor.cudaIsAvailable());
        info.put("device", device);
        
        return Response.ok(info).build();
    }
    
    /**
     * Initialize or reload model.
     */
    @POST
    @Path("/model/load")
    public Response loadModel() {
        try {
            if (model != null) {
                model.close();
            }
            
            model = new MNISTNet();
            model.eval();  // Set to evaluation mode
            
            // Move to GPU if available and enabled
            if (enableCuda && Tensor.cudaIsAvailable()) {
                model.to(Tensor.Device.CUDA);
            }
            
            Map<String, Object> result = new HashMap<>();
            result.put("status", "success");
            result.put("message", "Model loaded successfully");
            result.put("parameters", model.parameters().size());
            
            return Response.ok(result).build();
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                          .entity(error)
                          .build();
        }
    }
    
    /**
     * Perform inference on input data.
     * 
     * Request body: {"data": [784 float values]}
     * Response: {"prediction": digit, "probabilities": [10 values]}
     */
    @POST
    @Path("/predict")
    public Response predict(PredictionRequest request) {
        if (model == null) {
            loadModel();
        }
        
        try {
            // Validate input
            if (request.data == null || request.data.length != 784) {
                Map<String, String> error = new HashMap<>();
                error.put("error", "Input must be 784 values (28x28 image)");
                return Response.status(Response.Status.BAD_REQUEST)
                              .entity(error)
                              .build();
            }
            
            // Create tensor from input
            long[] shape = new long[]{1, 784};
            Tensor input = Tensor.zeros(shape, Tensor.ScalarType.FLOAT);
            
            // TODO: Copy data from request.data to tensor
            // This would use FFM to write to tensor's data pointer
            
            // Forward pass
            Tensor output = model.forward(input);
            Tensor probs = output.softmax(1);
            
            // Get prediction (would need argmax implementation)
            int prediction = 0;  // Placeholder
            double[] probabilities = new double[10];  // Placeholder
            
            // Cleanup
            input.close();
            output.close();
            probs.close();
            
            // Prepare response
            Map<String, Object> result = new HashMap<>();
            result.put("prediction", prediction);
            result.put("probabilities", probabilities);
            result.put("device", device);
            
            return Response.ok(result).build();
            
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                          .entity(error)
                          .build();
        }
    }
    
    /**
     * Batch prediction endpoint.
     */
    @POST
    @Path("/predict/batch")
    public Response predictBatch(BatchPredictionRequest request) {
        if (model == null) {
            loadModel();
        }
        
        try {
            int batchSize = request.batch.length;
            long[] shape = new long[]{batchSize, 784};
            
            Tensor input = Tensor.zeros(shape, Tensor.ScalarType.FLOAT);
            
            // TODO: Copy batch data to tensor
            
            Tensor output = model.forward(input);
            Tensor probs = output.softmax(1);
            
            // Extract predictions for each sample
            int[] predictions = new int[batchSize];  // Placeholder
            
            input.close();
            output.close();
            probs.close();
            
            Map<String, Object> result = new HashMap<>();
            result.put("predictions", predictions);
            result.put("batch_size", batchSize);
            
            return Response.ok(result).build();
            
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                          .entity(error)
                          .build();
        }
    }
    
    /**
     * Create random test tensor.
     */
    @POST
    @Path("/tensor/random")
    public Response createRandomTensor(TensorRequest request) {
        try {
            Tensor tensor = Tensor.randn(request.shape, Tensor.ScalarType.FLOAT);
            
            Map<String, Object> result = new HashMap<>();
            result.put("shape", Arrays.toString(tensor.shape()));
            result.put("numel", tensor.numel());
            result.put("dim", tensor.dim());
            
            tensor.close();
            
            return Response.ok(result).build();
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                          .entity(error)
                          .build();
        }
    }
    
    /**
     * Benchmark tensor operations.
     */
    @GET
    @Path("/benchmark")
    public Response benchmark() {
        Map<String, Object> results = new HashMap<>();
        
        try {
            // Matrix multiplication benchmark
            long start = System.nanoTime();
            Tensor a = Tensor.randn(new long[]{1000, 1000}, Tensor.ScalarType.FLOAT);
            Tensor b = Tensor.randn(new long[]{1000, 1000}, Tensor.ScalarType.FLOAT);
            Tensor c = a.matmul(b);
            long matmulTime = System.nanoTime() - start;
            
            results.put("matmul_ms", matmulTime / 1_000_000.0);
            
            a.close();
            b.close();
            c.close();
            
            // Element-wise operations benchmark
            start = System.nanoTime();
            Tensor x = Tensor.randn(new long[]{10000, 100}, Tensor.ScalarType.FLOAT);
            Tensor y = x.relu().sigmoid().tanh();
            long activationTime = System.nanoTime() - start;
            
            results.put("activation_chain_ms", activationTime / 1_000_000.0);
            
            x.close();
            y.close();
            
            results.put("device", device);
            results.put("cuda_available", Tensor.cudaIsAvailable());
            
            return Response.ok(results).build();
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("error", e.getMessage());
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                          .entity(error)
                          .build();
        }
    }
    
    // ========================================================================
    // Request/Response DTOs
    // ========================================================================
    
    public static class PredictionRequest {
        public float[] data;
    }
    
    public static class BatchPredictionRequest {
        public float[][] batch;
    }
    
    public static class TensorRequest {
        public long[] shape;
    }
}
