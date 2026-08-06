package tech.kayys.gollek.provider.litert;

import org.junit.platform.engine.discovery.DiscoverySelectors;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;
import org.junit.platform.launcher.listeners.SummaryGeneratingListener;
import org.junit.platform.launcher.listeners.TestExecutionSummary;

import java.io.PrintWriter;

public final class LiteRTRegressionMain {

    private LiteRTRegressionMain() {
    }

    public static void main(String[] args) {
        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(
                        DiscoverySelectors.selectClass(LiteRTGemmaNativeRunnerHeuristicsTest.class),
                        DiscoverySelectors.selectMethod(
                                "tech.kayys.gollek.provider.litert.LiteRTProviderTest#providerIsDiscoverableBeforeConfigInjection"),
                        DiscoverySelectors.selectMethod(
                                "tech.kayys.gollek.provider.litert.LiteRTGemmaDebugTest#gemma4TaskRunnerRejectsSlowRepeatedTokenPathByDefault"),
                        DiscoverySelectors.selectMethod(
                                "tech.kayys.gollek.provider.litert.LiteRTGemmaDebugTest#nativeRunnerRejectsRawLiteRtLmWhenDisabled"))
                .build();

        Launcher launcher = LauncherFactory.create();
        SummaryGeneratingListener listener = new SummaryGeneratingListener();
        launcher.registerTestExecutionListeners(listener);
        launcher.execute(request);

        TestExecutionSummary summary = listener.getSummary();
        PrintWriter out = new PrintWriter(System.out, true);
        summary.printTo(out);
        if (!summary.getFailures().isEmpty()) {
            out.println();
            out.println("LiteRT regression failures:");
            summary.getFailures().forEach(failure -> {
                out.println("- " + failure.getTestIdentifier().getDisplayName());
                failure.getException().printStackTrace(out);
            });
        }
        if (summary.getTestsFoundCount() == 0) {
            throw new IllegalStateException("No LiteRT regression tests were discovered");
        }
        if (summary.getTestsFailedCount() > 0) {
            throw new IllegalStateException("LiteRT regression tests failed: " + summary.getTestsFailedCount());
        }
    }
}
