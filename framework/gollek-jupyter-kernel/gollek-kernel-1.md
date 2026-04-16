package tech.kayys.gollek.jupyter;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.zeromq.ZMQ;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Main Jupyter Kernel for Gollek AI/ML Platform
 * Supports GPU-accelerated tensor operations, neural networks, and multimodal AI
 */
public class GollekKernel {
    private final GollekSession session;
    private final TensorDisplay tensorDisplay;
    private final CompletionProvider completer;
    private final Gson gson;
    private final AtomicInteger executionCount;
    private final ExecutorService executorService;
    
    // Jupyter Protocol Sockets
    private final ZMQ.Socket shellSocket;
    private final ZMQ.Socket iopubSocket;
    private final ZMQ.Socket stdinSocket;
    private final ZMQ.Socket controlSocket;
    private final ZMQ.Socket heartbeatSocket;
    private final ZMQ.Context context;
    
    private volatile boolean running = true;
    
    public GollekKernel(String connectionFile) throws Exception {
        this.gson = new Gson();
        this.executionCount = new AtomicInteger(0);
        this.executorService = Executors.newCachedThreadPool();
        
        // Load connection configuration
        JsonObject config = loadConnectionConfig(connectionFile);
        
        // Initialize ZeroMQ context
        this.context = ZMQ.context(1);
        
        // Create sockets
        this.shellSocket = context.createSocket(SocketType.ROUTER);
        this.iopubSocket = context.createSocket(SocketType.PUB);
        this.stdinSocket = context.createSocket(SocketType.ROUTER);
        this.controlSocket = context.createSocket(SocketType.ROUTER);
        this.heartbeatSocket = context.createSocket(SocketType.REP);
        
        // Bind sockets
        String transport = config.get("transport").getAsString();
        String ip = config.get("ip").getAsString();
        
        shellSocket.bind(String.format("%s://%s:%d", transport, ip, 
            config.get("shell_port").getAsInt()));
        iopubSocket.bind(String.format("%s://%s:%d", transport, ip,
            config.get("iopub_port").getAsInt()));
        stdinSocket.bind(String.format("%s://%s:%d", transport, ip,
            config.get("stdin_port").getAsInt()));
        controlSocket.bind(String.format("%s://%s:%d", transport, ip,
            config.get("control_port").getAsInt()));
        heartbeatSocket.bind(String.format("%s://%s:%d", transport, ip,
            config.get("hb_port").getAsInt()));
        
        // Initialize Gollek components
        this.session = new GollekSession();
        this.tensorDisplay = new TensorDisplay();
        this.completer = new CompletionProvider(session);
        
        // Send kernel info ready message
        sendKernelInfoReady();
        
        System.out.println("✅ Gollek Jupyter Kernel initialized");
        System.out.println("   Java version: " + System.getProperty("java.version"));
        System.out.println("   GPU support: " + session.isGPUAvailable());
    }
    
    private JsonObject loadConnectionConfig(String connectionFile) throws Exception {
        String content = new String(java.nio.file.Files.readAllBytes(
            java.nio.file.Paths.get(connectionFile)));
        return gson.fromJson(content, JsonObject.class);
    }
    
    private void sendKernelInfoReady() {
        JsonObject message = new JsonObject();
        message.addProperty("status", "ok");
        
        JsonObject kernelInfo = new JsonObject();
        kernelInfo.addProperty("protocol_version", "5.3");
        kernelInfo.addProperty("implementation", "gollek-jupyter-kernel");
        kernelInfo.addProperty("implementation_version", "1.0.0");
        kernelInfo.addProperty("language_name", "java");
        kernelInfo.addProperty("language_version", "25");
        
        JsonObject languageInfo = new JsonObject();
        languageInfo.addProperty("name", "java");
        languageInfo.addProperty("version", "25");
        languageInfo.addProperty("mimetype", "text/x-java-source");
        languageInfo.addProperty("file_extension", ".java");
        kernelInfo.add("language_info", languageInfo);
        
        JsonObject helpLinks = new JsonObject();
        helpLinks.addProperty("Gollek Documentation", "https://gollek-ai.github.io");
        helpLinks.addProperty("Java Documentation", "https://docs.oracle.com/en/java/javase/25/");
        kernelInfo.add("help_links", helpLinks);
        
        message.add("kernel_info", kernelInfo);
        
        sendIOPubMessage("kernel_info", message);
    }
    
    public void run() {
        // Start heartbeat responder
        Thread heartbeatThread = new Thread(this::handleHeartbeat);
        heartbeatThread.setDaemon(true);
        heartbeatThread.start();
        
        // Main shell message loop
        while (running) {
            try {
                byte[] message = shellSocket.recv(0);
                if (message != null) {
                    handleShellMessage(message);
                }
            } catch (Exception e) {
                System.err.println("Error in shell loop: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private void handleHeartbeat() {
        while (running) {
            try {
                byte[] request = heartbeatSocket.recv(0);
                if (request != null) {
                    heartbeatSocket.send(request, 0);
                }
            } catch (Exception e) {
                // Ignore heartbeat errors
            }
        }
    }
    
    private void handleShellMessage(byte[] messageData) {
        try {
            String[] frames = decodeFrames(messageData);
            if (frames.length < 5) return;
            
            String idents = frames[0];
            String hmac = frames[1];
            String header = frames[2];
            String parentHeader = frames[3];
            String metadata = frames[4];
            String content = frames.length > 5 ? frames[5] : "{}";
            
            JsonObject headerObj = gson.fromJson(header, JsonObject.class);
            String msgType = headerObj.get("msg_type").getAsString();
            
            switch (msgType) {
                case "kernel_info_request":
                    handleKernelInfoRequest(idents, header, parentHeader, metadata);
                    break;
                case "execute_request":
                    handleExecuteRequest(content, idents, header, parentHeader, metadata);
                    break;
                case "complete_request":
                    handleCompleteRequest(content, idents, header, parentHeader, metadata);
                    break;
                case "inspect_request":
                    handleInspectRequest(content, idents, header, parentHeader, metadata);
                    break;
                case "shutdown_request":
                    handleShutdownRequest(content, idents, header, parentHeader, metadata);
                    break;
            }
        } catch (Exception e) {
            System.err.println("Error handling message: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void handleExecuteRequest(String content, String idents, 
                                      String header, String parentHeader, String metadata) {
        JsonObject contentObj = gson.fromJson(content, JsonObject.class);
        String code = contentObj.get("code").getAsString();
        boolean silent = contentObj.has("silent") && contentObj.get("silent").getAsBoolean();
        int execCount = executionCount.incrementAndGet();
        
        // Send execution status
        sendIOPubMessage("status", createStatusMessage("busy", execCount));
        
        // Execute the code
        ExecutionResult result = null;
        try {
            result = executeCode(code, execCount);
        } catch (Exception e) {
            result = new ExecutionResult(null, e.getMessage(), true);
        }
        
        // Send execution results
        if (!silent) {
            if (result.error) {
                sendError(result.errorMessage, execCount, parentHeader);
            } else if (result.output != null) {
                sendExecutionResult(result.output, execCount, parentHeader);
            }
        }
        
        // Send execution status
        sendIOPubMessage("status", createStatusMessage("idle", execCount));
        
        // Send execute reply
        sendShellMessage(idents, "execute_reply", createExecuteReply(execCount, !result.error), 
                         header, parentHeader, metadata);
    }
    
    private ExecutionResult executeCode(String code, int execCount) {
        try {
            // Try evaluating as Gollek code
            Object result = session.evaluate(code);
            
            if (result == null) {
                return new ExecutionResult(null, null, false);
            }
            
            // Format output based on type
            String formatted = session.formatOutput(result);
            return new ExecutionResult(formatted, null, false);
            
        } catch (Exception e) {
            return new ExecutionResult(null, e.getMessage(), true);
        }
    }
    
    private void sendExecutionResult(String output, int execCount, String parentHeader) {
        JsonObject content = new JsonObject();
        content.addProperty("execution_count", execCount);
        
        JsonObject data = new JsonObject();
        data.addProperty("text/plain", output);
        content.add("data", data);
        
        JsonObject metadata = new JsonObject();
        content.add("metadata", metadata);
        
        sendIOPubMessage("execute_result", content);
    }
    
    private void sendError(String errorMessage, int execCount, String parentHeader) {
        JsonObject content = new JsonObject();
        content.addProperty("execution_count", execCount);
        content.addProperty("ename", "GollekException");
        content.addProperty("evalue", errorMessage);
        
        List<String> traceback = new ArrayList<>();
        traceback.add("Error in Gollek execution: " + errorMessage);
        content.add("traceback", gson.toJsonTree(traceback));
        
        sendIOPubMessage("error", content);
    }
    
    private void handleCompleteRequest(String content, String idents, 
                                       String header, String parentHeader, String metadata) {
        JsonObject contentObj = gson.fromJson(content, JsonObject.class);
        String code = contentObj.get("code").getAsString();
        int cursorPos = contentObj.get("cursor_pos").getAsInt();
        
        List<String> matches = completer.getCompletions(code, cursorPos);
        
        JsonObject replyContent = new JsonObject();
        replyContent.addProperty("cursor_start", completer.getCursorStart());
        replyContent.addProperty("cursor_end", completer.getCursorEnd());
        replyContent.add("matches", gson.toJsonTree(matches));
        replyContent.addProperty("metadata", new JsonObject());
        
        sendShellMessage(idents, "complete_reply", replyContent, header, parentHeader, metadata);
    }
    
    private void handleInspectRequest(String content, String idents,
                                      String header, String parentHeader, String metadata) {
        JsonObject contentObj = gson.fromJson(content, JsonObject.class);
        String code = contentObj.get("code").getAsString();
        int cursorPos = contentObj.get("cursor_pos").getAsInt();
        
        String doc = completer.getDocumentation(code, cursorPos);
        
        JsonObject replyContent = new JsonObject();
        replyContent.addProperty("found", doc != null);
        if (doc != null) {
            JsonObject data = new JsonObject();
            data.addProperty("text/plain", doc);
            data.addProperty("text/html", "<pre>" + doc + "</pre>");
            replyContent.add("data", data);
            replyContent.addProperty("metadata", new JsonObject());
        }
        
        sendShellMessage(idents, "inspect_reply", replyContent, header, parentHeader, metadata);
    }
    
    private void handleKernelInfoRequest(String idents, String header, 
                                         String parentHeader, String metadata) {
        JsonObject content = new JsonObject();
        content.addProperty("protocol_version", "5.3");
        content.addProperty("implementation", "gollek-jupyter-kernel");
        content.addProperty("implementation_version", "1.0.0");
        content.addProperty("language_name", "java");
        
        JsonObject languageInfo = new JsonObject();
        languageInfo.addProperty("name", "java");
        languageInfo.addProperty("version", "25");
        languageInfo.addProperty("mimetype", "text/x-java-source");
        languageInfo.addProperty("file_extension", ".java");
        
        JsonObject helpLinks = new JsonObject();
        helpLinks.addProperty("Gollek Documentation", "https://gollek-ai.github.io");
        languageInfo.add("help_links", helpLinks);
        
        content.add("language_info", languageInfo);
        
        sendShellMessage(idents, "kernel_info_reply", content, header, parentHeader, metadata);
    }
    
    private void handleShutdownRequest(String content, String idents,
                                       String header, String parentHeader, String metadata) {
        JsonObject contentObj = gson.fromJson(content, JsonObject.class);
        boolean restart = contentObj.get("restart").getAsBoolean();
        
        running = false;
        
        JsonObject replyContent = new JsonObject();
        replyContent.addProperty("restart", restart);
        
        sendShellMessage(idents, "shutdown_reply", replyContent, header, parentHeader, metadata);
        
        cleanup();
    }
    
    private void sendShellMessage(String idents, String msgType, JsonObject content,
                                  String header, String parentHeader, String metadata) {
        JsonObject msgHeader = createMessageHeader(msgType);
        String msgId = msgHeader.get("msg_id").getAsString();
        
        List<String> frames = new ArrayList<>();
        frames.add(idents);
        frames.add(""); // HMAC placeholder
        frames.add(msgHeader.toString());
        frames.add(parentHeader != null ? parentHeader : "{}");
        frames.add(metadata != null ? metadata : "{}");
        frames.add(content.toString());
        
        byte[] message = encodeFrames(frames);
        shellSocket.send(message, 0);
    }
    
    private void sendIOPubMessage(String msgType, JsonObject content) {
        JsonObject header = createMessageHeader(msgType);
        String msgId = header.get("msg_id").getAsString();
        
        List<String> frames = new ArrayList<>();
        frames.add(""); // ident
        frames.add(""); // HMAC placeholder
        frames.add(header.toString());
        frames.add("{}"); // parent header
        frames.add("{}"); // metadata
        frames.add(content.toString());
        
        byte[] message = encodeFrames(frames);
        iopubSocket.send(message, 0);
    }
    
    private JsonObject createMessageHeader(String msgType) {
        JsonObject header = new JsonObject();
        header.addProperty("msg_id", UUID.randomUUID().toString());
        header.addProperty("username", "gollek_kernel");
        header.addProperty("session", UUID.randomUUID().toString());
        header.addProperty("msg_type", msgType);
        header.addProperty("version", "5.3");
        header.addProperty("date", new java.util.Date().toInstant().toString());
        return header;
    }
    
    private JsonObject createStatusMessage(String status, int execCount) {
        JsonObject statusMsg = new JsonObject();
        statusMsg.addProperty("execution_state", status);
        return statusMsg;
    }
    
    private JsonObject createExecuteReply(int execCount, boolean success) {
        JsonObject reply = new JsonObject();
        reply.addProperty("status", success ? "ok" : "error");
        reply.addProperty("execution_count", execCount);
        return reply;
    }
    
    private String[] decodeFrames(byte[] data) {
        String str = new String(data);
        // Simple delimiter-based frame decoding (ZMQ multipart)
        return str.split("\0");
    }
    
    private byte[] encodeFrames(List<String> frames) {
        return String.join("\0", frames).getBytes();
    }
    
    private void cleanup() {
        executorService.shutdown();
        session.close();
        shellSocket.close();
        iopubSocket.close();
        stdinSocket.close();
        controlSocket.close();
        heartbeatSocket.close();
        context.close();
    }
    
    private static class ExecutionResult {
        final String output;
        final String errorMessage;
        final boolean error;
        
        ExecutionResult(String output, String errorMessage, boolean error) {
            this.output = output;
            this.errorMessage = errorMessage;
            this.error = error;
        }
    }
}