package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import io.quarkus.arc.Unremovable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Convenience command to stop Gollek API server (alias for "gollek server stop")
 */
@Dependent
@Unremovable
@Command(name = "stop",
        description = "Stop the running Gollek API server (shorthand for 'gollek server stop')")
public class StopCommand implements Runnable {

    @Option(names = {"--port"}, description = "Port the server is running on (default: 9131)")
    int port = 9131;

    @Override
    public void run() {
        try {
            ServerCommand.StopSubcommand subcommand = new ServerCommand.StopSubcommand();
            subcommand.port = this.port;
            subcommand.run();
        } catch (Exception e) {
            System.err.println("Error stopping server: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
