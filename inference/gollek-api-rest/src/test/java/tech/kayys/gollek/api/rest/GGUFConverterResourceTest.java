package tech.kayys.gollek.api.rest;

import jakarta.ws.rs.BadRequestException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.gollek.converter.ConversionStorageService;
import tech.kayys.gollek.converter.GGUFConverter;
import tech.kayys.gollek.converter.dto.ConversionRequest;
import tech.kayys.gollek.converter.dto.ConversionResponse;
import tech.kayys.gollek.converter.GGUFException;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.QuantizationType;
import tech.kayys.gollek.spi.context.RequestContext;

import java.nio.file.Path;
import io.smallrye.mutiny.Uni;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class GGUFConverterResourceTest {

    @Test
    void previewConversionAllowsCommunityAbsolutePaths() {
        GGUFConverterResource resource = new GGUFConverterResource();
        resource.converter = Mockito.mock(GGUFConverter.class);
        resource.storageService = Mockito.mock(ConversionStorageService.class);
        resource.requestContext = null;

        when(resource.converter.resolveModelBasePath()).thenReturn(Path.of("/tmp/models"));
        when(resource.converter.resolveConverterBasePath()).thenReturn(Path.of("/tmp/conversions"));

        GGUFConversionParams resolved = GGUFConversionParams.builder()
                .inputPath(Path.of("/tmp/model"))
                .outputPath(Path.of("/tmp/out.gguf"))
                .quantization(QuantizationType.Q4_K_M)
                .build();
        when(resource.converter.resolveParams(any())).thenReturn(resolved);

        ConversionRequest request = ConversionRequest.builder()
                .inputPath("/tmp/model")
                .outputPath("/tmp/out.gguf")
                .quantization(QuantizationType.Q4_K_M)
                .build();

        Response response = resource.previewConversion(request);
        assertThat(response.getStatus()).isEqualTo(200);
        ConversionResponse payload = (ConversionResponse) response.getEntity();
        assertThat(payload.isDryRun()).isTrue();
        assertThat(payload.getInputPath()).isEqualTo("/tmp/model");
        assertThat(payload.getOutputPath()).isEqualTo("/tmp/out.gguf");
        assertThat(payload.getInputBasePath()).isNotBlank();
        assertThat(payload.getOutputBasePath()).isNotBlank();
        assertThat(payload.getDerivedOutputName()).isEqualTo("out.gguf");
    }

    @Test
    void previewConversionRejectsAbsolutePathsForTenants() {
        GGUFConverterResource resource = new GGUFConverterResource();
        resource.converter = Mockito.mock(GGUFConverter.class);
        resource.storageService = Mockito.mock(ConversionStorageService.class);
        resource.requestContext = Mockito.mock(RequestContext.class);
        when(resource.requestContext.getRequestId()).thenReturn("tenant1");

        ConversionRequest request = ConversionRequest.builder()
                .inputPath("/tmp/model")
                .outputPath("out.gguf")
                .quantization(QuantizationType.Q4_K_M)
                .build();

        assertThrows(BadRequestException.class, () -> resource.previewConversion(request));
    }

    @Test
    void convertModelDryRunReturnsResolvedPaths() {
        GGUFConverterResource resource = new GGUFConverterResource();
        resource.converter = Mockito.mock(GGUFConverter.class);
        resource.storageService = Mockito.mock(ConversionStorageService.class);
        resource.requestContext = null;

        when(resource.converter.resolveModelBasePath()).thenReturn(Path.of("/tmp/models"));
        when(resource.converter.resolveConverterBasePath()).thenReturn(Path.of("/tmp/conversions"));

        GGUFConversionParams resolved = GGUFConversionParams.builder()
                .inputPath(Path.of("/tmp/model"))
                .outputPath(Path.of("/tmp/out.gguf"))
                .quantization(QuantizationType.Q4_K_M)
                .build();
        when(resource.converter.resolveParams(any())).thenReturn(resolved);

        ConversionRequest request = ConversionRequest.builder()
                .inputPath("/tmp/model")
                .outputPath("/tmp/out.gguf")
                .quantization(QuantizationType.Q4_K_M)
                .dryRun(true)
                .build();

        ConversionResponse response = resource.convertModel(request).await().indefinitely();
        assertThat(response.isDryRun()).isTrue();
        assertThat(response.getInputPath()).isEqualTo("/tmp/model");
        assertThat(response.getOutputPath()).isEqualTo("/tmp/out.gguf");
    }

    @Test
    void convertModelPropagatesOverwriteErrors() {
        GGUFConverterResource resource = new GGUFConverterResource();
        resource.converter = Mockito.mock(GGUFConverter.class);
        resource.storageService = Mockito.mock(ConversionStorageService.class);
        resource.requestContext = null;

        when(resource.converter.resolveModelBasePath()).thenReturn(Path.of("/tmp/models"));
        when(resource.converter.resolveConverterBasePath()).thenReturn(Path.of("/tmp/conversions"));

        when(resource.converter.convertAsync(any()))
                .thenReturn(Uni.createFrom().failure(new GGUFException("Output path already exists")));

        ConversionRequest request = ConversionRequest.builder()
                .inputPath("/tmp/model")
                .outputPath("/tmp/out.gguf")
                .quantization(QuantizationType.Q4_K_M)
                .build();

        assertThrows(GGUFException.class, () -> resource.convertModel(request).await().indefinitely());
    }
}
