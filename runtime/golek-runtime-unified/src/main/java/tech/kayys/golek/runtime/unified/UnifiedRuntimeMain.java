package tech.kayys.golek.runtime.unified;

import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.annotations.QuarkusMain;
import picocli.CommandLine;

/**
 * Main entry point for the unified Golek runtime.
 * This combines the CLI, REST API, and Web UI in a single executable.
 */
@QuarkusMain
public class UnifiedRuntimeMain {

    public static void main(String[] args) {
        // Check if we're running in CLI mode or server mode
        boolean isCliMode = args.length > 0 && !args[0].startsWith("--");
        
        if (isCliMode) {
            // Run CLI command
            System.exit(runCli(args));
        } else {
            // Start the server (REST API + Web UI)
            Quarkus.run(args);
        }
    }

    private static int runCli(String[] args) {
        // Initialize and run the CLI
        return new CommandLine(new tech.kayys.golek.cli.GolekCommand()).execute(args);
    }
}