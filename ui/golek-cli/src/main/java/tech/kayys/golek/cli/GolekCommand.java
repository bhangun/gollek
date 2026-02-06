package tech.kayys.golek.cli;

import io.quarkus.picocli.runtime.annotations.TopCommand;
import picocli.CommandLine.Command;
import tech.kayys.golek.cli.commands.ChatCommand;
import tech.kayys.golek.cli.commands.InfoCommand;
import tech.kayys.golek.cli.commands.ListCommand;
import tech.kayys.golek.cli.commands.ProvidersCommand;
import tech.kayys.golek.cli.commands.PullCommand;
import tech.kayys.golek.cli.commands.RunCommand;
import tech.kayys.golek.cli.commands.ShowCommand;

@TopCommand
@Command(name = "golek", mixinStandardHelpOptions = true, version = "1.0.0", description = "Golek Inference CLI - Run local and cloud AI models", subcommands = {
                RunCommand.class,
                ChatCommand.class,
                PullCommand.class,
                ListCommand.class,
                ShowCommand.class,
                ProvidersCommand.class,
                InfoCommand.class
})
public class GolekCommand {
}
