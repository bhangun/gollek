package tech.kayys.golek.cli.commands;

import picocli.CommandLine.Command;

@Command(name = "info", description = "Display system info and available adapters")
public class InfoCommand implements Runnable {

    @Override
    public void run() {
        System.out.println("Golek Inference CLI v1.0.0");
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("OS: " + System.getProperty("os.name"));
        // TODO: Inject ModelRegistryService to list actual models/adapters later
        System.out.println("Adapter: Default (GGUF)");
    }
}
