package tech.kayys.golek.cli;

import io.quarkus.picocli.runtime.annotations.TopCommand;
import picocli.CommandLine.Command;
import tech.kayys.golek.cli.commands.ChatCommand;
import tech.kayys.golek.cli.commands.ExtensionsCommand;
import tech.kayys.golek.cli.commands.InfoCommand;
import tech.kayys.golek.cli.commands.ListCommand;
import tech.kayys.golek.cli.commands.ProvidersCommand;
import tech.kayys.golek.cli.commands.PullCommand;
import tech.kayys.golek.cli.commands.RunCommand;
import tech.kayys.golek.cli.commands.ShowCommand;

import picocli.CommandLine;
import picocli.CommandLine.Option;

@TopCommand
@Command(name = "golek", mixinStandardHelpOptions = true, version = "1.0.0", description = "Golek Inference CLI - Run local and cloud AI models", subcommands = {
        RunCommand.class,
        ChatCommand.class,
        PullCommand.class,
        ListCommand.class,
        ShowCommand.class,
        ProvidersCommand.class,
        ExtensionsCommand.class,
        InfoCommand.class
})
public class GolekCommand implements Runnable {

    @Option(names = { "--log" }, description = "Enable verbose logging", scope = CommandLine.ScopeType.INHERIT)
    boolean verbose;

    public GolekCommand() {
    }

    @Override
    public void run() {
        if (verbose) {
            System.setProperty("quarkus.log.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.golek\".level", "DEBUG");
            System.setProperty("gguf.provider.verbose-logging", "true");
            // Workaround for programmatic change during runtime if possible,
            // but Picocli runs before Quarkus finishes all init in some modes.
            // For now, these system properties might help, or we check this flag in
            // commands.
        }
    }
}
