package tech.kayys.gollek.mcp.registry;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import tech.kayys.gollek.sdk.core.exception.SdkException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class McpRegistryEngineModeTest {

    @Test
    void remoteModeRequiresEnterprise() {
        McpRegistryEngine engine = new McpRegistryEngine();
        withProperty("gollek.mcp.registry.mode", "remote", () -> {
            withProperty("gollek.enterprise.enabled", "false", () -> {
                SdkException exception = assertThrows(SdkException.class, engine::list);
                assertNotNull(exception.getCause());
                assertTrue(exception.getCause().getMessage().contains("requires enterprise mode"));
            });
        });
    }

    @Test
    void enterpriseRemoteModeSendsTenantHeader() throws Exception {
        AtomicReference<String> tenantHeader = new AtomicReference<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/api/v1/mcp/registry/servers", exchange -> {
            tenantHeader.set(exchange.getRequestHeaders().getFirst("X-Tenant-Id"));
            byte[] body = "[]".getBytes();
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        int port = server.getAddress().getPort();

        try {
            McpRegistryEngine engine = new McpRegistryEngine();
            withProperty("gollek.mcp.registry.mode", "remote", () -> {
                withProperty("gollek.enterprise.enabled", "true", () -> {
                    withProperty("gollek.mcp.registry.api.base-url", "http://localhost:" + port, () -> {
                        withProperty("gollek.mcp.registry.tenant-header.enabled", "true", () -> {
                            withProperty("gollek.tenant.id", "tenant-enterprise-01", () -> {
                                List<?> result = engine.list();
                                assertEquals(0, result.size());
                            });
                        });
                    });
                });
            });
            assertEquals("tenant-enterprise-01", tenantHeader.get());
        } finally {
            server.stop(0);
        }
    }

    private void withProperty(String key, String value, ThrowingRunnable action) {
        String old = System.getProperty(key);
        try {
            if (value == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, value);
            }
            action.run();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            if (old == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, old);
            }
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
