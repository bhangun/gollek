package tech.kayys.wayang.tool.dto;

import java.util.Map;

public record HttpRequestContext(
                HttpMethod method,
                String url,
                Map<String, String> queryParams,
                Map<String, String> headers,
                Object body,
                String contentType) {
}
