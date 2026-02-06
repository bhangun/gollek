package com.kmpchatbot.server;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import static io.restassured.RestAssured.given;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.notNullValue;

@QuarkusTest
public class ChatResourceTest {
    
    @Test
    public void testHealthEndpoint() {
        given()
            .when().get("/api/chat/health")
            .then()
                .statusCode(200)
                .body("status", is("Chat service is running"));
    }
    
    @Test
    public void testGetConversations() {
        given()
            .when().get("/api/chat/conversations")
            .then()
                .statusCode(200)
                .contentType(ContentType.JSON);
    }
    
    @Test
    public void testSendMessage() {
        String requestBody = """
            {
                "message": "Hello, this is a test message"
            }
            """;
        
        given()
            .contentType(ContentType.JSON)
            .body(requestBody)
            .when().post("/api/chat/message")
            .then()
                .statusCode(200)
                .body("message", notNullValue())
                .body("session_id", notNullValue())
                .body("role", is("assistant"));
    }
    
    @Test
    public void testInvalidRequest() {
        String requestBody = """
            {
                "message": ""
            }
            """;
        
        given()
            .contentType(ContentType.JSON)
            .body(requestBody)
            .when().post("/api/chat/message")
            .then()
                .statusCode(400);
    }
}
