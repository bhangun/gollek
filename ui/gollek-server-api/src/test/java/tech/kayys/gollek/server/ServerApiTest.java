package tech.kayys.gollek.server;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.RestAssured;
import org.junit.jupiter.api.Test;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

@QuarkusTest
public class ServerApiTest {

    @Test
    public void testHealth() {
        RestAssured.given()
                .when().get("/health")
                .then().statusCode(200)
                .body("status", equalTo("ok"));
    }

    @Test
    public void testListModels() {
        RestAssured.given().header("X-API-Key", "community")
                .when().get("/v1/models")
                .then().statusCode(200)
                .body("size()", equalTo(0));
    }
}
