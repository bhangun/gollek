/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationResourceTest.java
 * ───────────────────────
 * Tests for QuantizationResource REST API.
 */
package tech.kayys.gollek.safetensor.quantization.rest;

import io.quarkus.test.junit.QuarkusTest;
import io.restassured.http.ContentType;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.*;

/**
 * Integration tests for QuantizationResource REST API.
 */
@QuarkusTest
class QuantizationResourceTest {

    @Test
    void testListStrategies() {
        given()
                .contentType(ContentType.JSON)
                .when().get("/api/v1/quantization/strategies")
                .then()
                .statusCode(200)
                .body("strategies", hasSize(3))
                .body("strategies[0].name", equalTo("INT4"))
                .body("strategies[1].name", equalTo("INT8"))
                .body("strategies[2].name", equalTo("FP8"));
    }

    @Test
    void testRecommendWithSmallModel() {
        given()
                .contentType(ContentType.JSON)
                .queryParam("model_size_gb", 2.0)
                .queryParam("prioritize_quality", true)
                .when().get("/api/v1/quantization/recommend")
                .then()
                .statusCode(200)
                .body("recommended_strategy", notNullValue())
                .body("model_size_gb", equalTo(2.0))
                .body("prioritize_quality", equalTo(true));
    }

    @Test
    void testRecommendWithLargeModel() {
        given()
                .contentType(ContentType.JSON)
                .queryParam("model_size_gb", 13.0)
                .queryParam("prioritize_quality", false)
                .when().get("/api/v1/quantization/recommend")
                .then()
                .statusCode(200)
                .body("recommended_strategy", equalTo("INT4"))
                .body("model_size_gb", equalTo(13.0));
    }

    @Test
    void testQuantizeWithMissingInputPath() {
        Map<String, Object> request = Map.of(
                "output_path", "/tmp/output",
                "strategy", "INT4");

        given()
                .contentType(ContentType.JSON)
                .body(request)
                .when().post("/api/v1/quantization/quantize")
                .then()
                .statusCode(400)
                .body("success", equalTo(false))
                .body("error_message", containsString("input_path"));
    }

    @Test
    void testQuantizeWithInvalidStrategy() {
        Map<String, Object> request = Map.of(
                "input_path", "/tmp/model",
                "output_path", "/tmp/output",
                "strategy", "INVALID");

        given()
                .contentType(ContentType.JSON)
                .body(request)
                .when().post("/api/v1/quantization/quantize")
                .then()
                .statusCode(400)
                .body("success", equalTo(false))
                .body("error_message", containsString("Invalid strategy"));
    }

    @Test
    void testQuantizeWithMissingOutputPath() {
        Map<String, Object> request = Map.of(
                "input_path", "/tmp/model",
                "strategy", "INT4");

        given()
                .contentType(ContentType.JSON)
                .body(request)
                .when().post("/api/v1/quantization/quantize")
                .then()
                .statusCode(400)
                .body("success", equalTo(false))
                .body("error_message", containsString("output_path"));
    }
}
