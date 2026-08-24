package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import io.quarkus.arc.Unremovable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.sdk.util.GollekHome;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Manage Gollek API server lifecycle.
 * Supports running the server in background, foreground with debug output, and stopping.
 */
@Dependent
@Unremovable
@Command(name = "server",
        description = "Manage Gollek API server lifecycle",
        subcommands = {ServerCommand.ServeSubcommand.class, ServerCommand.StopSubcommand.class})
public class ServerCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Override
    public void run() {
        // Show help if no subcommand provided
        System.out.println("Usage: gollek server <command> [options]");
        System.out.println("Commands:");
        System.out.println("  start [--debug]     Start the Gollek API server (background by default, use --debug for foreground)");
        System.out.println("  stop                Stop the running Gollek API server");
    }

    /**
     * Start the server (background or foreground with --debug)
     */
    @Command(name = "start", description = "Start the Gollek API server")
    @Dependent
    @Unremovable
    public static class ServeSubcommand implements Runnable {

        @ParentCommand
        ServerCommand serverCommand;

        @Option(names = {"--debug"}, description = "Run server in foreground with TTY output (not backgrounded)")
        boolean debug = false;

        @Option(names = {"--port"}, description = "Port to run the server on (default: 9131)")
        int port = 9131;

        @Option(names = {"--grpc"}, description = "Only serve gRPC")
        boolean grpcOnly = false;

        @Option(names = {"--rest"}, description = "Only serve REST")
        boolean restOnly = false;

        @Override
        public void run() {
            try {
                if (debug) {
                    // Run in foreground with TTY output
                    runServerForeground(port);
                } else {
                    // Run in background
                    runServerBackground(port);
                }
            } catch (Exception e) {
                System.err.println("Error starting server: " + e.getMessage());
                e.printStackTrace();
                System.exit(1);
            }
        }

        private void runServerForeground(int port) throws IOException, InterruptedException {
            System.out.println("🚀 Starting Gollek API server on port " + port + " (foreground mode)");
            System.out.println("Press Ctrl-C to stop");
            System.out.println("");

            ProcessBuilder pb = buildServerProcess(port);
            pb.inheritIO(); // Inherit parent's stdin/stdout/stderr for live output
            
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            System.out.println("");
            System.out.println("Server stopped with exit code: " + exitCode);
        }

        private void runServerBackground(int port) throws IOException {
            // Check if server is already running
            if (isServerRunning(port)) {
                System.out.println("⚠️  Server is already running on port " + port);
                System.out.println("To stop it, run: gollek server stop");
                System.out.println("To start in debug mode, run: gollek server start --debug");
                return;
            }

            System.out.println("🚀 Starting Gollek API server on port " + port + " (background mode)");
            
            // Create PID file directory
            Path pidDir = GollekHome.path("server");
            Files.createDirectories(pidDir);
            Path pidFile = pidDir.resolve("server.pid");

            // Start process
            ProcessBuilder pb = buildServerProcess(port);
            
            // Redirect output to logs directory with rotation strategy
            Path logDir = GollekHome.path("server", "logs");
            Files.createDirectories(logDir);
            
            // Rotate old server.log to dated backup if it exists and is from a previous day
            Path currentLogFile = logDir.resolve("server.log");
            if (Files.exists(currentLogFile)) {
                rotateLogIfNeeded(currentLogFile, logDir);
            }
            
            // Always append to server.log (current/live log)
            File logFile = currentLogFile.toFile();
            
            pb.redirectOutput(ProcessBuilder.Redirect.appendTo(logFile));
            pb.redirectError(ProcessBuilder.Redirect.appendTo(logFile));
            
            // Start the process
            Process process = pb.start();
            int pid = (int) process.pid();
            
            // Write PID file
            Files.writeString(pidFile, String.valueOf(pid));
            
            System.out.println("✅ Server started with PID " + pid);
            System.out.println("📝 Current logs: " + logFile.getAbsolutePath());
            System.out.println("");
            System.out.println("To stop the server, run: gollek server stop");
            System.out.println("To view live logs: tail -f " + logFile.getAbsolutePath());
        }
        
        /**
         * Rotate log file if it's from a previous day
         */
        private void rotateLogIfNeeded(Path currentLogFile, Path logDir) throws IOException {
            try {
                // Get the last modified date of the current log file
                long lastModified = Files.getLastModifiedTime(currentLogFile).toMillis();
                LocalDateTime fileDate = LocalDateTime.ofInstant(
                    java.time.Instant.ofEpochMilli(lastModified),
                    java.time.ZoneId.systemDefault()
                );
                LocalDateTime today = LocalDateTime.now();
                
                // If the file is from a different day, rotate it
                if (!fileDate.toLocalDate().equals(today.toLocalDate())) {
                    String dateStr = fileDate.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
                    Path backupFile = logDir.resolve("server-" + dateStr + ".log");
                    
                    // If backup already exists, append to it; otherwise rename
                    if (Files.exists(backupFile)) {
                        Files.write(backupFile, Files.readAllBytes(currentLogFile), 
                            java.nio.file.StandardOpenOption.APPEND);
                    } else {
                        Files.move(currentLogFile, backupFile);
                    }
                }
            } catch (Exception e) {
                // If rotation fails, continue anyway (log rotation is non-critical)
                System.err.println("Warning: Failed to rotate log file: " + e.getMessage());
            }
        }

        private ProcessBuilder buildServerProcess(int port) {
            // Find the gollek-api jar
            String jarPath = findGollekApiJar();
            if (jarPath == null) {
                throw new RuntimeException("Gollek API server JAR not found. Please build the project first.");
            }

            ProcessBuilder pb = new ProcessBuilder(
                    "java",
                    "-jar",
                    jarPath,
                    "-Dquarkus.http.port=" + (grpcOnly ? "-1" : String.valueOf(port))
            );

            // Set environment variable for port
            pb.environment().put("QUARKUS_HTTP_PORT", grpcOnly ? "-1" : String.valueOf(port));
            
            // Handle --grpc and --rest
            if (grpcOnly) {
                pb.environment().put("QUARKUS_GRPC_SERVER_ENABLE", "true");
            } else if (restOnly) {
                pb.environment().put("QUARKUS_GRPC_SERVER_ENABLE", "false");
            }

            return pb;
        }

        private String findGollekApiJar() {
            // Try common locations where gradle/maven build artifacts are placed
            String[] jarLocations = {
                    // Gradle locations
                    "gollek/ui/gollek-api/build/quarkus-app/quarkus-run.jar",
                    "gollek/ui/gollek-api/target/quarkus-app/quarkus-run.jar",
                    "ui/gollek-api/build/quarkus-app/quarkus-run.jar",
                    "ui/gollek-api/target/quarkus-app/quarkus-run.jar",
                    // Alternative gradle locations
                    "gollek/ui/gollek-api/build/libs/gollek-server-runtime-0.1.0-SNAPSHOT-runner.jar",
                    "gollek/ui/gollek-api/target/gollek-server-runtime-0.1.0-SNAPSHOT-runner.jar",
            };

            // Search relative to current directory
            for (String location : jarLocations) {
                File f = new File(location);
                if (f.exists() && f.isFile()) {
                    return f.getAbsolutePath();
                }
            }

            // Search relative to parent directories (common CI/CD scenario)
            for (String location : jarLocations) {
                File f = new File("../" + location);
                if (f.exists() && f.isFile()) {
                    return f.getAbsolutePath();
                }
                f = new File("../../" + location);
                if (f.exists() && f.isFile()) {
                    return f.getAbsolutePath();
                }
            }

            return null;
        }

        private boolean isServerRunning(int port) {
            try {
                ProcessBuilder pb = new ProcessBuilder("lsof", "-i", ":" + port);
                Process p = pb.start();
                int exitCode = p.waitFor();
                return exitCode == 0;
            } catch (Exception e) {
                // lsof might not be available; try netstat
                try {
                    ProcessBuilder pb = new ProcessBuilder("netstat", "-ln");
                    Process p = pb.start();
                    String output = new String(p.getInputStream().readAllBytes());
                    return output.contains(":" + port);
                } catch (Exception ex) {
                    // If we can't determine, assume not running
                    return false;
                }
            }
        }
    }

    /**
     * Stop the running server
     */
    @Command(name = "stop", description = "Stop the Gollek API server")
    @Dependent
    @Unremovable
    public static class StopSubcommand implements Runnable {

        @ParentCommand
        ServerCommand serverCommand;

        @Option(names = {"--port"}, description = "Port the server is running on (default: 9131)")
        int port = 9131;

        @Override
        public void run() {
            try {
                stopServer(port);
            } catch (Exception e) {
                System.err.println("Error stopping server: " + e.getMessage());
                e.printStackTrace();
                System.exit(1);
            }
        }

        private void stopServer(int port) throws IOException, InterruptedException {
            // Try to read PID from file
            Path pidFile = GollekHome.path("server", "server.pid");
            
            if (pidFile.toFile().exists()) {
                try {
                    String pidStr = Files.readString(pidFile).trim();
                    int pid = Integer.parseInt(pidStr);
                    
                    // Kill the process
                    if (isProcessRunning(pid)) {
                        System.out.println("Stopping server (PID: " + pid + ")...");
                        killProcess(pid);
                        Thread.sleep(1000);
                        
                        // Verify it stopped
                        if (!isProcessRunning(pid)) {
                            System.out.println("✅ Server stopped successfully");
                            Files.deleteIfExists(pidFile);
                        } else {
                            System.out.println("⚠️  Server still running, forcing kill...");
                            forceKillProcess(pid);
                            Thread.sleep(500);
                            if (!isProcessRunning(pid)) {
                                System.out.println("✅ Server force-killed successfully");
                                Files.deleteIfExists(pidFile);
                            } else {
                                System.out.println("❌ Failed to kill server process");
                            }
                        }
                    } else {
                        System.out.println("⚠️  Server process (PID: " + pid + ") is not running");
                        Files.deleteIfExists(pidFile);
                    }
                } catch (NumberFormatException e) {
                    System.err.println("❌ Invalid PID in " + pidFile);
                }
            } else {
                // Try to find and kill by port
                System.out.println("Looking for server on port " + port + "...");
                boolean killed = killProcessByPort(port);
                if (killed) {
                    System.out.println("✅ Server stopped successfully");
                } else {
                    System.out.println("⚠️  No server found running on port " + port);
                }
            }
        }

        private boolean killProcessByPort(int port) throws IOException, InterruptedException {
            // Use lsof to find process by port
            try {
                ProcessBuilder pb = new ProcessBuilder("lsof", "-i", ":" + port, "-t");
                Process p = pb.start();
                String pidStr = new String(p.getInputStream().readAllBytes()).trim();
                int exitCode = p.waitFor();
                
                if (exitCode == 0 && !pidStr.isEmpty()) {
                    int pid = Integer.parseInt(pidStr.split("\\n")[0]);
                    killProcess(pid);
                    Thread.sleep(500);
                    return !isProcessRunning(pid);
                }
            } catch (Exception e) {
                // lsof might not be available, try fuser
                try {
                    ProcessBuilder pb = new ProcessBuilder("fuser", port + "/tcp", "-k");
                    int exitCode = pb.start().waitFor();
                    return exitCode == 0;
                } catch (Exception ex) {
                    // fuser also failed
                }
            }
            return false;
        }

        private boolean isProcessRunning(int pid) throws IOException, InterruptedException {
            ProcessBuilder pb = new ProcessBuilder("ps", "-p", String.valueOf(pid));
            Process p = pb.start();
            return p.waitFor() == 0;
        }

        private void killProcess(int pid) throws IOException {
            new ProcessBuilder("kill", String.valueOf(pid)).start();
        }

        private void forceKillProcess(int pid) throws IOException {
            new ProcessBuilder("kill", "-9", String.valueOf(pid)).start();
        }
    }
}
