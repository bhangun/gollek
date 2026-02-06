package com.kmpchatbot.server.config;

import com.kmpchatbot.server.domain.User;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.transaction.Transactional;
import org.jboss.logging.Logger;

@ApplicationScoped
public class DataInitializer {

    private static final Logger LOG = Logger.getLogger(DataInitializer.class);

    @Transactional
    public void loadData(@Observes StartupEvent event) {
        LOG.info("Initializing database with sample data...");

        // Check if default user exists
        if (User.findByUsername("demo") == null) {
            User demoUser = new User();
            demoUser.setUsername("demo");
            demoUser.setPassword("demo123"); // In production, hash this!
            demoUser.setEmail("demo@kmpchatbot.com");
            demoUser.setFullName("Demo User");
            demoUser.setActive(true);
            demoUser.persist();
            
            LOG.info("Created demo user (username: demo, password: demo123)");
        }

        // Create admin user if doesn't exist
        if (User.findByUsername("admin") == null) {
            User adminUser = new User();
            adminUser.setUsername("admin");
            adminUser.setPassword("admin123"); // In production, hash this!
            adminUser.setEmail("admin@kmpchatbot.com");
            adminUser.setFullName("Admin User");
            adminUser.setActive(true);
            adminUser.persist();
            
            LOG.info("Created admin user (username: admin, password: admin123)");
        }

        LOG.info("Database initialization complete!");
    }
}
