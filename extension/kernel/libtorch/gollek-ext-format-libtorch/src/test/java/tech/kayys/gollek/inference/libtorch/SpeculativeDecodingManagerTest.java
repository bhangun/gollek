package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.gollek.inference.libtorch.core.Tensor;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class SpeculativeDecodingManagerTest {

        @Mock
        LibTorchSessionManager sessionManager;

        @Mock
        ContinuousBatchingManager batchingManager;

        @Mock
        TorchScriptRunner runner;

        @Mock
        LibTorchSessionManager.SessionContext sessionContext;

        @InjectMocks
        SpeculativeDecodingManager manager;

        @BeforeEach
        void setup() {
                // Setup session manager mock for DRAFT model only
                lenient().when(sessionManager.getSession(anyString(), anyString(), any()))
                                .thenReturn(sessionContext);
                lenient().when(sessionContext.runner()).thenReturn(runner);
        }

        @Test
        void testExecuteSpeculativeStep_AcceptAll() {
                // Setup scenarios
                String tenantId = "tenant1";
                String draftModelId = "draft-model";
                Path draftPath = Path.of("draft.pt");
                String targetModelId = "target-model";
                Path targetPath = Path.of("target.pt");
                long[] promptIds = { 1, 2, 3 };
                // DEFAULT_LOOKAHEAD = 5 in code, let's assume it generates 5 drafts.
                // Or if we can't change it, we must match it. Code says 5.

                // Mock Tensor static methods
                try (MockedStatic<Tensor> mockedTensor = Mockito.mockStatic(Tensor.class)) {
                        // Mock Tensor instances
                        Tensor inputTensor = mock(Tensor.class);
                        Tensor logitsTensor = mock(Tensor.class);
                        Tensor argmaxTensor = mock(Tensor.class);

                        // Mock Tensor.fromLongArray (for draft generation) and fromFloatArray (for
                        // verification)
                        mockedTensor.when(() -> Tensor.fromLongArray(any(long[].class), any(long[].class)))
                                        .thenReturn(inputTensor);
                        mockedTensor.when(() -> Tensor.fromFloatArray(any(float[].class), any(long[].class)))
                                        .thenReturn(inputTensor); // reuse mock

                        // Mock Draft Execution flow (5 iterations)
                        when(runner.forward(any(Tensor.class))).thenReturn(logitsTensor);
                        when(logitsTensor.argmax(anyLong())).thenReturn(argmaxTensor);

                        // Draft outputs: 4, 5, 6, 7, 8
                        when(argmaxTensor.toLongArray())
                                        .thenReturn(new long[] { 0, 0, 4 }) // i=0
                                        .thenReturn(new long[] { 0, 0, 0, 5 }) // i=1
                                        .thenReturn(new long[] { 0, 0, 0, 0, 6 }) // i=2
                                        .thenReturn(new long[] { 0, 0, 0, 0, 0, 7 }) // i=3
                                        .thenReturn(new long[] { 0, 0, 0, 0, 0, 0, 8 }); // i=4

                        // Mock Verification via BatchingManager
                        when(batchingManager.enqueue(any(ContinuousBatchingManager.BatchRequest.class)))
                                        .thenAnswer(invocation -> {
                                                ContinuousBatchingManager.BatchRequest req = invocation.getArgument(0);

                                                // Handler invocation needs a mocked tensor that returns verification
                                                // results
                                                Tensor verifySlice = mock(Tensor.class);
                                                Tensor verifyArgmax = mock(Tensor.class);
                                                when(verifySlice.argmax(-1)).thenReturn(verifyArgmax);
                                                // Expected verification output: prompt(3) + draft(5) -> 8 tokens input
                                                // Output prediction should match draft: 4, 5, 6, 7, 8
                                                // Note: Code checks predArray[pos].
                                                // Input [1,2,3, 4,5,6,7,8]
                                                // pos for first draft check: prompt.len + i - 1 = 3 + 0 - 1 = 2 (index
                                                // of last prompt token '3')
                                                // prediction at pos 2 should be '4' (first draft token).

                                                // So toLongArray should return array where:
                                                // arr[2] = 4
                                                // arr[3] = 5
                                                // arr[4] = 6
                                                // arr[5] = 7
                                                // arr[6] = 8

                                                when(verifyArgmax.toLongArray())
                                                                .thenReturn(new long[] { 0, 0, 4, 5, 6, 7, 8, 9 });

                                                InferenceResponse resp = req.handler().apply(verifySlice);
                                                req.future().complete(resp);
                                                return req.future();
                                        });

                        // Execute
                        int[] accepted = manager.executeSpeculativeStep(tenantId, draftModelId, draftPath,
                                        targetModelId,
                                        targetPath, promptIds);

                        // Assertions
                        assertThat(accepted).containsExactly(4, 5, 6, 7, 8);

                        // Verify session release (only 1 context acquired/released for draft)
                        verify(sessionManager, times(1)).releaseSession(eq(tenantId), eq(draftModelId), any());
                }
        }

        @Test
        void testExecuteSpeculativeStep_RejectSome() {
                String tenantId = "tenant1";
                String draftModelId = "draft-model";
                Path draftPath = Path.of("draft.pt");
                String targetModelId = "target-model";
                Path targetPath = Path.of("target.pt");
                long[] promptIds = { 1, 2, 3 };

                try (MockedStatic<Tensor> mockedTensor = Mockito.mockStatic(Tensor.class)) {
                        Tensor inputTensor = mock(Tensor.class);
                        Tensor logitsTensor = mock(Tensor.class);
                        Tensor argmaxTensor = mock(Tensor.class);

                        mockedTensor.when(() -> Tensor.fromLongArray(any(long[].class), any(long[].class)))
                                        .thenReturn(inputTensor);
                        mockedTensor.when(() -> Tensor.fromFloatArray(any(float[].class), any(long[].class)))
                                        .thenReturn(inputTensor);

                        when(runner.forward(any(Tensor.class))).thenReturn(logitsTensor);
                        when(logitsTensor.argmax(anyLong())).thenReturn(argmaxTensor);

                        // Drafts: 4, 5, 6, 7, 8 (same as accept case)
                        when(argmaxTensor.toLongArray())
                                        .thenReturn(new long[] { 0, 0, 4 })
                                        .thenReturn(new long[] { 0, 0, 0, 5 })
                                        .thenReturn(new long[] { 0, 0, 0, 0, 6 })
                                        .thenReturn(new long[] { 0, 0, 0, 0, 0, 7 })
                                        .thenReturn(new long[] { 0, 0, 0, 0, 0, 0, 8 });

                        // Verify with mismatch
                        when(batchingManager.enqueue(any(ContinuousBatchingManager.BatchRequest.class)))
                                        .thenAnswer(invocation -> {
                                                ContinuousBatchingManager.BatchRequest req = invocation.getArgument(0);
                                                Tensor verifySlice = mock(Tensor.class);
                                                Tensor verifyArgmax = mock(Tensor.class);
                                                when(verifySlice.argmax(-1)).thenReturn(verifyArgmax);

                                                // Verification input prompt len 3. Drafts start at index 3 (pos 2 in
                                                // array for prediction).
                                                // Expect:
                                                // pos 2 -> 4 (Match draft 4)
                                                // pos 3 -> 9 (Mismatch! Draft was 5)
                                                // Rest doesn't matter as we stop.

                                                when(verifyArgmax.toLongArray())
                                                                .thenReturn(new long[] { 0, 0, 4, 9, 8, 8, 8, 8 });

                                                InferenceResponse resp = req.handler().apply(verifySlice);
                                                req.future().complete(resp);
                                                return req.future();
                                        });

                        int[] accepted = manager.executeSpeculativeStep(tenantId, draftModelId, draftPath,
                                        targetModelId,
                                        targetPath, promptIds);

                        // Expect: 4 (match), 9 (bonus)
                        assertThat(accepted).containsExactly(4, 9);
                }
        }
}
