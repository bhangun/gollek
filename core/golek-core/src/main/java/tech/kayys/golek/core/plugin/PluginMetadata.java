/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author bhangun
 */

package tech.kayys.golek.core.plugin;

/**
 * Plugin metadata
 */
public class PluginMetadata {
    private final String id;
    private final String name;
    private final String version;
    private final String description;
    private final String vendor;
    private final String homepage;
    private final String license;
    private final String[] authors;

    public PluginMetadata(String id, String name, String version, String description,
            String vendor, String homepage, String license, String[] authors) {
        this.id = id;
        this.name = name;
        this.version = version;
        this.description = description;
        this.vendor = vendor;
        this.homepage = homepage;
        this.license = license;
        this.authors = authors != null ? authors.clone() : new String[0];
    }

    // Getters
    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getDescription() {
        return description;
    }

    public String getVendor() {
        return vendor;
    }

    public String getHomepage() {
        return homepage;
    }

    public String getLicense() {
        return license;
    }

    public String[] getAuthors() {
        return authors.clone();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String id;
        private String name;
        private String version;
        private String description;
        private String vendor;
        private String homepage;
        private String license;
        private String[] authors;

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder vendor(String vendor) {
            this.vendor = vendor;
            return this;
        }

        public Builder homepage(String homepage) {
            this.homepage = homepage;
            return this;
        }

        public Builder license(String license) {
            this.license = license;
            return this;
        }

        public Builder authors(String... authors) {
            this.authors = authors;
            return this;
        }

        public PluginMetadata build() {
            return new PluginMetadata(id, name, version, description, vendor, homepage, license, authors);
        }
    }
}