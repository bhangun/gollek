package tech.kayys.gollek.cli.audit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import tech.kayys.gollek.cli.GollekHome;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Persistence service for quantization audit records.
 * <p>
 * Writes audit trails to {@code ~/.gollek/audit/} in both JSON and CSV formats.
 * Supports appending to a cumulative CSV file for spreadsheet analysis.
 */
public class QuantAuditTrail {

    private static final Path AUDIT_DIR = GollekHome.path("audit");
    private static final Path CSV_FILE = AUDIT_DIR.resolve("quantization-audit.csv");
    private static final DateTimeFormatter TS_FMT = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

    private static final ObjectMapper JSON = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .enable(SerializationFeature.INDENT_OUTPUT)
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

    /**
     * Persist an audit record as JSON and append to the cumulative CSV.
     *
     * @param record the audit record to persist
     * @return the path to the written JSON file
     */
    public static Path persist(QuantAuditRecord record) throws IOException {
        Files.createDirectories(AUDIT_DIR);

        // Write individual JSON file
        String timestamp = LocalDateTime.now().format(TS_FMT);
        String jsonFilename = "quant-" + timestamp + "-" + record.strategy() + ".json";
        Path jsonPath = AUDIT_DIR.resolve(jsonFilename);
        JSON.writeValue(jsonPath.toFile(), record);

        // Append to cumulative CSV
        appendCsv(record);

        return jsonPath;
    }

    /**
     * Persist an audit record to a custom path.
     */
    public static Path persist(QuantAuditRecord record, Path customPath) throws IOException {
        Files.createDirectories(customPath.getParent());
        JSON.writeValue(customPath.toFile(), record);
        appendCsv(record);
        return customPath;
    }

    /**
     * Append audit record to the cumulative CSV file.
     * Creates the file with headers if it doesn't exist yet.
     */
    private static synchronized void appendCsv(QuantAuditRecord record) throws IOException {
        boolean writeHeader = !Files.exists(CSV_FILE);
        try (var writer = Files.newBufferedWriter(CSV_FILE,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (writeHeader) {
                writer.write(QuantAuditRecord.csvHeader());
                writer.newLine();
            }
            writer.write(record.toCsvLine());
            writer.newLine();
        }
    }

    /**
     * Read all audit records from the audit directory.
     */
    public static List<QuantAuditRecord> readAll() throws IOException {
        List<QuantAuditRecord> records = new ArrayList<>();
        if (!Files.isDirectory(AUDIT_DIR)) return records;

        try (var files = Files.list(AUDIT_DIR)) {
            files.filter(p -> p.toString().endsWith(".json"))
                 .sorted()
                 .forEach(p -> {
                     try {
                         records.add(JSON.readValue(p.toFile(), QuantAuditRecord.class));
                     } catch (Exception ignored) {
                         // skip malformed entries
                     }
                 });
        }
        return records;
    }

    /**
     * Export all audit records to a CSV file at the given path.
     */
    public static void exportCsv(Path outputPath) throws IOException {
        List<QuantAuditRecord> records = readAll();
        Files.createDirectories(outputPath.getParent());
        try (var writer = Files.newBufferedWriter(outputPath)) {
            writer.write(QuantAuditRecord.csvHeader());
            writer.newLine();
            for (QuantAuditRecord r : records) {
                writer.write(r.toCsvLine());
                writer.newLine();
            }
        }
    }

    /**
     * Get the default audit directory path.
     */
    public static Path auditDir() {
        return AUDIT_DIR;
    }

    /**
     * Get the cumulative CSV file path.
     */
    public static Path csvFile() {
        return CSV_FILE;
    }
}
