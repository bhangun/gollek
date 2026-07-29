package tech.kayys.wayang.yaffffm;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.lang.reflect.Method;
import java.lang.foreign.MemorySegment;

/**
 * Minimal control-plane and lifecycle manager for YAFF SHM frames.
 *
 * - Allocates temp files in /dev/shm (or java.io.tmpdir) and memory-maps them
 * - Writes provided bytes into the mapped buffer
 * - Stores a durable reference (metadata) in UnifiedMemoryStore (RocksDB) under
 * key "yaff:frame:{id}"
 * - Exposes HTTP endpoints: /allocate (POST with raw bytes) and /release (POST
 * with JSON {id})
 *
 * This is intentionally small and self-contained for prototype verification.
 */
public class ShmLifecycleManager {

    private final WayangYaffShmTransport transport;
    private final Object store; // reflective UnifiedMemoryStore or null
    private final Method storePutMethod;
    private final Method storeDeleteMethod;
    private final Method storeFlushMethod;

    private final ObjectMapper mapper = new ObjectMapper();

    private final HttpServer server;

    public ShmLifecycleManager(int port) throws IOException {
        this.transport = new WayangYaffShmTransport(WayangYaffShmTransport.defaultShmDir());
        // Try to open UnifiedMemoryStore via reflection; if not available, fallback to
        // local file metadata
        Object tmpStore = null;
        Method putM = null;
        Method delM = null;
        Method flushM = null;
        try {
            String dbPath = System.getProperty("wayang.alkhawarizm.dbpath", "./data/alkhawarizm");
            Class<?> factoryCls = Class.forName("tech.kayys.alkhawarizm.core.memory.MemoryStoreFactory");
            Method open = factoryCls.getMethod("openRocksDb", String.class);
            tmpStore = open.invoke(null, dbPath);
            Class<?> storeCls = tmpStore.getClass();
            putM = storeCls.getMethod("put", byte[].class, java.lang.foreign.MemorySegment.class);
            delM = storeCls.getMethod("delete", byte[].class);
            flushM = storeCls.getMethod("flush");
        } catch (ClassNotFoundException cnf) {
            tmpStore = null;
        } catch (Throwable t) {
            tmpStore = null;
        }
        this.store = tmpStore;
        this.storePutMethod = putM;
        this.storeDeleteMethod = delM;
        this.storeFlushMethod = flushM;

        server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) Executors.newFixedThreadPool(4);
        server.setExecutor(threadPoolExecutor);

        server.createContext("/allocate", new AllocateHandler());
        server.createContext("/release", new ReleaseHandler());
        server.start();
    }

    public void shutdown() {
        server.stop(1);
        if (store != null && storeFlushMethod != null) {
            try {
                storeFlushMethod.invoke(store);
            } catch (Throwable ignored) {
            }
        }
    }

    public record FrameMeta(String id, String path, long offset, long length) {
    }

    private class AllocateHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            byte[] body = exchange.getRequestBody().readAllBytes();
            // allocate shm file
            MappedByteBuffer mb = null;
            Path backing = null;
            try {
                backing = WayangYaffShmTransport.defaultShmDir()
                        .resolve("yaff-" + UUID.randomUUID().toString() + ".dat");
                FileChannel fc = FileChannel.open(backing, StandardOpenOption.CREATE_NEW, StandardOpenOption.READ,
                        StandardOpenOption.WRITE);
                fc.truncate(body.length);
                mb = fc.map(FileChannel.MapMode.READ_WRITE, 0, body.length);
                mb.put(body);
                mb.force();

                String id = UUID.randomUUID().toString();
                FrameMeta meta = new FrameMeta(id, backing.toString(), 0L, body.length);
                // persist metadata as JSON in store if available, otherwise write to local file
                byte[] metaBytes = mapper.writeValueAsBytes(Map.of(
                        "id", id,
                        "path", meta.path,
                        "offset", meta.offset,
                        "length", meta.length));
                if (store != null && storePutMethod != null) {
                    try {
                        MemorySegment seg = MemorySegment.ofArray(metaBytes);
                        storePutMethod.invoke(store, ("yaff:frame:" + id).getBytes(), seg);
                    } catch (Throwable t) {
                        Files.write(WayangYaffShmTransport.defaultShmDir().resolve("yaff-meta-" + id + ".json"),
                                metaBytes);
                    }
                } else {
                    Files.write(WayangYaffShmTransport.defaultShmDir().resolve("yaff-meta-" + id + ".json"), metaBytes);
                }

                byte[] resp = mapper.writeValueAsBytes(
                        Map.of("id", id, "path", meta.path, "offset", meta.offset, "length", meta.length));
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, resp.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(resp);
                }
            } catch (Throwable t) {
                exchange.sendResponseHeaders(500, 0);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(("error:" + t.getMessage()).getBytes());
                }
            }
        }
    }

    private class ReleaseHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            byte[] body = exchange.getRequestBody().readAllBytes();
            Map m = mapper.readValue(body, Map.class);
            String id = (String) m.get("id");
            if (id == null) {
                exchange.sendResponseHeaders(400, 0);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write("missing id".getBytes());
                }
                return;
            }
            try {
                if (store != null && storeDeleteMethod != null) {
                    try {
                        storeDeleteMethod.invoke(store, ("yaff:frame:" + id).getBytes());
                        exchange.sendResponseHeaders(200, 0);
                        exchange.getResponseBody().close();
                        return;
                    } catch (Throwable t) {
                        // fallthrough to file removal
                    }
                }
                // remove local metadata file if present
                Path metaFile = WayangYaffShmTransport.defaultShmDir().resolve("yaff-meta-" + id + ".json");
                try {
                    Files.deleteIfExists(metaFile);
                    exchange.sendResponseHeaders(200, 0);
                    exchange.getResponseBody().close();
                    return;
                } catch (Throwable t) {
                }
                exchange.sendResponseHeaders(500, 0);
                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().close();
            } catch (Throwable t) {
                exchange.sendResponseHeaders(500, 0);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(("error:" + t.getMessage()).getBytes());
                }
            }
        }
    }
}
