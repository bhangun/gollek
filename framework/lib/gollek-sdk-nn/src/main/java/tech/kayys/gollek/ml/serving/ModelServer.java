package tech.kayys.gollek.ml.serving;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;

/**
 * Lightweight HTTP model serving server — exposes a trained model as a REST endpoint.
 *
 * <p>Listens on a configurable port and handles:
 * <ul>
 *   <li>{@code POST /predict} — runs model inference on a JSON float array input</li>
 *   <li>{@code GET  /health}  — returns {@code {"status":"ok"}}</li>
 *   <li>{@code GET  /info}    — returns model parameter count and class name</li>
 * </ul>
 *
 * <p>Uses JDK 25 virtual threads via {@link Executors#newVirtualThreadPerTaskExecutor()}
 * for non-blocking concurrent request handling — no thread pool sizing needed.
 *
 * <h3>Request format ({@code POST /predict})</h3>
 * <pre>
 *   Content-Type: application/json
 *   Body: {"input": [1.0, 2.0, 3.0, ...]}
 * </pre>
 *
 * <h3>Response format</h3>
 * <pre>
 *   {"output": [0.1, 0.9, ...]}
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var server = ModelServer.builder()
 *     .model(trainedModel)
 *     .inputShape(new long[]{1, 784})
 *     .port(8080)
 *     .build();
 *
 * server.start();
 * // curl -X POST http://localhost:8080/predict -d '{"input":[0.1,0.2,...]}'
 * server.stop();
 * }</pre>
 */
public final class ModelServer {

    private final NNModule  model;
    private final long[]  inputShape;
    private final int     port;
    private ServerSocket  serverSocket;
    private volatile boolean running = false;

    private ModelServer(Builder b) {
        this.model      = b.model;
        this.inputShape = b.inputShape;
        this.port       = b.port;
    }

    /**
     * Starts the HTTP server in a background virtual thread.
     *
     * @throws IOException if the port cannot be bound
     */
    public void start() throws IOException {
        serverSocket = new ServerSocket(port);
        running = true;
        Thread.ofVirtual().name("model-server").start(this::acceptLoop);
        System.out.printf("ModelServer started on http://localhost:%d%n", port);
    }

    /**
     * Stops the server and closes the listening socket.
     *
     * @throws IOException if the socket cannot be closed
     */
    public void stop() throws IOException {
        running = false;
        if (serverSocket != null) serverSocket.close();
    }

    // ── Accept loop ───────────────────────────────────────────────────────

    private void acceptLoop() {
        try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    exec.submit(() -> handleRequest(client));
                } catch (IOException e) {
                    if (running) e.printStackTrace();
                }
            }
        }
    }

    private void handleRequest(Socket client) {
        try (client;
             var in  = new BufferedReader(new InputStreamReader(client.getInputStream(),  StandardCharsets.UTF_8));
             var out = new PrintWriter(new OutputStreamWriter(client.getOutputStream(), StandardCharsets.UTF_8), true)) {

            // Parse HTTP request line
            String requestLine = in.readLine();
            if (requestLine == null) return;
            String[] parts = requestLine.split(" ");
            String method = parts[0], path = parts.length > 1 ? parts[1] : "/";

            // Read headers to find Content-Length
            int contentLength = 0;
            String line;
            while ((line = in.readLine()) != null && !line.isEmpty()) {
                if (line.toLowerCase().startsWith("content-length:"))
                    contentLength = Integer.parseInt(line.split(":")[1].trim());
            }

            // Route
            String body = "", responseBody;
            if ("POST".equals(method) && contentLength > 0) {
                char[] buf = new char[contentLength];
                in.read(buf, 0, contentLength);
                body = new String(buf);
            }

            responseBody = switch (path) {
                case "/predict" -> handlePredict(body);
                case "/health"  -> "{\"status\":\"ok\"}";
                case "/info"    -> String.format("{\"model\":\"%s\",\"parameters\":%d}",
                                       model.getClass().getSimpleName(), model.parameterCount());
                default         -> "{\"error\":\"not found\"}";
            };

            int status = path.equals("/predict") || path.equals("/health") || path.equals("/info") ? 200 : 404;
            writeHttpResponse(out, status, responseBody);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Runs model inference on the JSON input array.
     *
     * @param body JSON body: {@code {"input": [...]}}
     * @return JSON response: {@code {"output": [...]}}
     */
    private String handlePredict(String body) {
        try {
            float[] input = parseFloatArray(body);
            GradTensor tensor = GradTensor.of(input, inputShape);
            model.eval();
            GradTensor output = model.forward(tensor);
            return "{\"output\":" + floatArrayToJson(output.data()) + "}";
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────

    private static void writeHttpResponse(PrintWriter out, int status, String body) {
        String statusText = status == 200 ? "OK" : "Not Found";
        out.print("HTTP/1.1 " + status + " " + statusText + "\r\n");
        out.print("Content-Type: application/json\r\n");
        out.print("Content-Length: " + body.getBytes(StandardCharsets.UTF_8).length + "\r\n");
        out.print("Connection: close\r\n");
        out.print("\r\n");
        out.print(body);
        out.flush();
    }

    /** Parses {@code {"input":[1.0,2.0,...]}} into a float array. */
    private static float[] parseFloatArray(String json) {
        int start = json.indexOf('[') + 1;
        int end   = json.lastIndexOf(']');
        String[] tokens = json.substring(start, end).split(",");
        float[] result = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++)
            result[i] = Float.parseFloat(tokens[i].trim());
        return result;
    }

    private static String floatArrayToJson(float[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(arr[i]);
        }
        return sb.append(']').toString();
    }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link ModelServer}.
     */
    public static final class Builder {
        private NNModule model;
        private long[] inputShape;
        private int    port = 8080;

        /** @param model trained model to serve */
        public Builder model(NNModule m)          { this.model = m; return this; }
        /** @param shape expected input shape (including batch dim) */
        public Builder inputShape(long... shape){ this.inputShape = shape; return this; }
        /** @param port HTTP port to listen on (default 8080) */
        public Builder port(int p)              { this.port = p; return this; }

        /**
         * Builds the {@link ModelServer}.
         *
         * @return configured server (not yet started)
         */
        public ModelServer build() { return new ModelServer(this); }
    }
}
