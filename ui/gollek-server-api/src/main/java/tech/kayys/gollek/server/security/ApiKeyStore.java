package tech.kayys.gollek.server.security;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.ObjectMapper;

@ApplicationScoped
public class ApiKeyStore {

    private static final Logger LOG = Logger.getLogger(ApiKeyStore.class);

    @ConfigProperty(name = "gollek.server.keys-file", defaultValue = "./data/keys.json")
    String keysFilePath;

    private final Set<String> keys = Collections.synchronizedSet(new HashSet<>());
    private final ObjectMapper mapper = new ObjectMapper();

    @PostConstruct
    void init() {
        Path p = Path.of(keysFilePath);
        try {
            if (Files.exists(p)) {
                var arr = mapper.readValue(p.toFile(), String[].class);
                for (String k : arr) keys.add(k);
            } else {
                // seed with community key if empty
                keys.add("community");
                persist();
            }
        } catch (IOException e) {
            LOG.warn("Failed to load keys file: " + e.getMessage());
            keys.add("community");
        }
    }

    public Set<String> listKeys() {
        return Set.copyOf(keys);
    }

    public boolean addKey(String key) {
        boolean added = keys.add(key);
        if (added) persist();
        return added;
    }

    public boolean removeKey(String key) {
        boolean removed = keys.remove(key);
        if (removed) persist();
        return removed;
    }

    private void persist() {
        Path p = Path.of(keysFilePath);
        try {
            Files.createDirectories(p.getParent());
            mapper.writeValue(p.toFile(), keys.toArray(new String[0]));
        } catch (IOException e) {
            LOG.warn("Failed to persist keys file: " + e.getMessage());
        }
    }
}
