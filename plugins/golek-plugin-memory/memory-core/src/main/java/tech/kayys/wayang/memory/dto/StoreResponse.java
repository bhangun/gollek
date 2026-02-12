
package tech.kayys.wayang.memory.dto;

public record StoreResponse(

        boolean success,
        String memoryId,
        String message) {
}