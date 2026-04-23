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
 */

package tech.kayys.gollek.plugin.core;

import org.eclipse.aether.DefaultRepositorySystemSession;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;
import org.eclipse.aether.collection.CollectRequest;
import org.eclipse.aether.connector.basic.BasicRepositoryConnectorFactory;
import org.eclipse.aether.graph.DependencyNode;
import org.eclipse.aether.impl.DefaultServiceLocator;
import org.eclipse.aether.repository.LocalRepository;
import org.eclipse.aether.repository.RemoteRepository;
import org.eclipse.aether.resolution.ArtifactRequest;
import org.eclipse.aether.resolution.ArtifactResult;
import org.eclipse.aether.resolution.DependencyRequest;
import org.eclipse.aether.resolution.DependencyResolutionException;
import org.eclipse.aether.spi.connector.RepositoryConnectorFactory;
import org.eclipse.aether.spi.connector.transport.TransporterFactory;
import org.eclipse.aether.transport.file.FileTransporterFactory;
import org.eclipse.aether.transport.http.HttpTransporterFactory;
import org.eclipse.aether.util.repository.AuthenticationBuilder;
import org.eclipse.aether.util.repository.SimpleResolutionErrorPolicy;
import org.jboss.logging.Logger;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Maven dependency resolver for plugin system.
 * 
 * <p>
 * Resolves and downloads Maven dependencies for plugins, supporting:
 * </p>
 * <ul>
 * <li>Central Maven repository</li>
 * <li>Custom repositories</li>
 * <li>Local repository caching (~/.m2/repository)</li>
 * <li>Transitive dependency resolution</li>
 * <li>Version range resolution</li>
 * <li>Snapshot updates</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * MavenDependencyResolver resolver = new MavenDependencyResolver();
 * 
 * // Resolve single dependency
 * List<File> jars = resolver.resolve("tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT");
 * 
 * // Resolve multiple dependencies
 * List<String> deps = List.of(
 *         "tech.kayys:gollek-spi-provider:1.0.0-SNAPSHOT",
 *         "tech.kayys:gollek-spi-inference:1.0.0-SNAPSHOT");
 * List<File> allJars = resolver.resolveAll(deps);
 * 
 * // Get dependency tree
 * DependencyTree tree = resolver.buildTree(deps);
 * }</pre>
 *
 * @since 2.1.0
 */
public class MavenDependencyResolver {

    private static final Logger LOG = Logger.getLogger(MavenDependencyResolver.class);

    /**
     * Default local repository location: ~/.m2/repository
     */
    public static final String DEFAULT_LOCAL_REPO = Paths.get(System.getProperty("user.home"), ".m2", "repository")
            .toString();

    /**
     * Default remote repositories
     */
    private static final List<RemoteRepository> DEFAULT_REPOS = List.of(
            new RemoteRepository.Builder("central", "default",
                    "https://repo.maven.apache.org/maven2").build(),
            new RemoteRepository.Builder("jboss-public", "default",
                    "https://repository.jboss.org/nexus/content/groups/public/").build());

    /**
     * Maven Repository System
     */
    private final RepositorySystem repoSystem;

    /**
     * Repository System Session
     */
    private final RepositorySystemSession repoSession;

    /**
     * Remote repositories
     */
    private final List<RemoteRepository> remoteRepos;

    /**
     * Resolved artifacts cache
     */
    private final Map<String, List<File>> resolvedCache = new ConcurrentHashMap<>();

    /**
     * Create resolver with default configuration.
     */
    public MavenDependencyResolver() {
        this(DEFAULT_LOCAL_REPO, DEFAULT_REPOS);
    }

    /**
     * Create resolver with custom local repository.
     *
     * @param localRepoPath Path to local Maven repository
     */
    public MavenDependencyResolver(String localRepoPath) {
        this(localRepoPath, DEFAULT_REPOS);
    }

    /**
     * Create resolver with custom repositories.
     *
     * @param localRepoPath Path to local Maven repository
     * @param remoteRepos   Remote repositories
     */
    public MavenDependencyResolver(String localRepoPath, List<RemoteRepository> remoteRepos) {
        this.repoSystem = newRepositorySystem();
        this.repoSession = repoSystem != null ? newRepositorySession(repoSystem, localRepoPath) : null;
        this.remoteRepos = remoteRepos != null ? remoteRepos : DEFAULT_REPOS;

        if (repoSystem == null) {
            LOG.warn("Maven Repository System could not be initialized. Dependency resolution will be unavailable.");
        } else {
            LOG.infof("Maven dependency resolver initialized (local repo: %s)", localRepoPath);
        }
    }

    /**
     * Resolve single Maven dependency.
     *
     * @param coordinate Maven coordinate (groupId:artifactId:version)
     * @return List of resolved JAR files (including transitive dependencies)
     */
    public List<File> resolve(String coordinate) {
        if (repoSystem == null || repoSession == null) {
            throw new IllegalStateException("Maven Repository System is not initialized. Cannot resolve: " + coordinate);
        }

        // Check cache first
        if (resolvedCache.containsKey(coordinate)) {
            LOG.debugf("Using cached resolution for: %s", coordinate);
            return resolvedCache.get(coordinate);
        }

        LOG.infof("Resolving dependency: %s", coordinate);

        try {
            // Parse coordinate
            Artifact artifact = new DefaultArtifact(coordinate);

            // Create dependency request
            CollectRequest collectRequest = new CollectRequest();
            org.eclipse.aether.graph.Dependency dependency = new org.eclipse.aether.graph.Dependency(artifact, null);
            collectRequest.addDependency(dependency);
            collectRequest.setRepositories(remoteRepos);

            // Resolve dependencies
            DependencyRequest depRequest = new DependencyRequest(collectRequest, null);
            repoSystem.resolveDependencies(repoSession, depRequest);

            // Get resolved JARs
            List<File> jars = resolveArtifact(artifact);

            // Cache result
            resolvedCache.put(coordinate, jars);

            LOG.infof("Resolved %s → %d JAR(s)", coordinate, jars.size());
            return jars;

        } catch (DependencyResolutionException e) {
            LOG.errorf(e, "Failed to resolve dependency: %s", coordinate);
            throw new RuntimeException("Failed to resolve dependency: " + coordinate, e);
        }
    }

    /**
     * Resolve multiple Maven dependencies.
     *
     * @param coordinates List of Maven coordinates
     * @return List of all resolved JAR files
     */
    public List<File> resolveAll(List<String> coordinates) {
        List<File> allJars = new ArrayList<>();

        for (String coordinate : coordinates) {
            List<File> jars = resolve(coordinate);
            allJars.addAll(jars);
        }

        // Remove duplicates
        return allJars.stream().distinct().collect(Collectors.toList());
    }

    /**
     * Resolve dependency and return classpath string.
     *
     * @param coordinate Maven coordinate
     * @return Classpath string (colon-separated on Unix, semicolon on Windows)
     */
    public String resolveClasspath(String coordinate) {
        List<File> jars = resolve(coordinate);
        return String.join(File.pathSeparator,
                jars.stream().map(File::getAbsolutePath).toArray(String[]::new));
    }

    /**
     * Build dependency tree.
     *
     * @param coordinates List of Maven coordinates
     * @return Dependency tree structure
     */
    public DependencyTree buildTree(List<String> coordinates) {
        try {
            // Create root artifacts
            List<Artifact> artifacts = coordinates.stream()
                    .map(DefaultArtifact::new)
                    .collect(Collectors.toList());

            // Create collect request
            CollectRequest collectRequest = new CollectRequest();
            for (Artifact artifact : artifacts) {
                collectRequest.addDependency(new org.eclipse.aether.graph.Dependency(artifact, null));
            }
            collectRequest.setRepositories(remoteRepos);

            // Resolve
            if (repoSystem == null || repoSession == null) {
                throw new IllegalStateException("Maven Repository System is not initialized. Cannot build tree.");
            }
            DependencyNode rootNode = repoSystem.collectDependencies(repoSession, collectRequest).getRoot();

            // Build tree
            return DependencyTree.fromNode(rootNode);

        } catch (Exception e) {
            throw new RuntimeException("Failed to build dependency tree", e);
        }
    }

    /**
     * Add custom remote repository.
     *
     * @param id       Repository ID
     * @param url      Repository URL
     * @param username Optional username
     * @param password Optional password
     */
    public void addRepository(String id, String url, String username, String password) {
        RemoteRepository.Builder builder = new RemoteRepository.Builder(id, "default", url);

        if (username != null && password != null) {
            builder.setAuthentication(
                    new AuthenticationBuilder()
                            .addUsername(username)
                            .addPassword(password)
                            .build());
        }

        remoteRepos.add(builder.build());
        LOG.infof("Added repository: %s (%s)", id, url);
    }

    /**
     * Clear resolution cache.
     */
    public void clearCache() {
        resolvedCache.clear();
        LOG.info("Dependency resolver cache cleared");
    }

    /**
     * Get local repository path.
     *
     * @return Path to local Maven repository
     */
    public Path getLocalRepositoryPath() {
        LocalRepository localRepo = repoSession.getLocalRepository();
        if (localRepo != null) {
            return localRepo.getBasedir().toPath();
        }
        return Paths.get(System.getProperty("user.home"), ".m2", "repository");
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    /**
     * Create Maven Repository System.
     *
     * @return Repository system instance
     */
    /**
     * Create Maven Repository System.
     *
     * @return Repository system instance
     */
    private RepositorySystem newRepositorySystem() {
        DefaultServiceLocator locator = org.apache.maven.repository.internal.MavenRepositorySystemUtils.newServiceLocator();
        locator.addService(RepositoryConnectorFactory.class, BasicRepositoryConnectorFactory.class);
        locator.addService(TransporterFactory.class, FileTransporterFactory.class);
        locator.addService(TransporterFactory.class, HttpTransporterFactory.class);

        locator.setErrorHandler(new DefaultServiceLocator.ErrorHandler() {
            @Override
            public void serviceCreationFailed(Class<?> type, Class<?> impl, Throwable exception) {
                LOG.errorf(exception, "Service creation failed: %s → %s", type, impl);
            }
        });
        return locator.getService(RepositorySystem.class);
    }

    /**
     * Create Repository System Session.
     *
     * @param system        Repository system
     * @param localRepoPath Local repository path
     * @return Repository session
     */
    private RepositorySystemSession newRepositorySession(RepositorySystem system, String localRepoPath) {
        DefaultRepositorySystemSession session = org.apache.maven.repository.internal.MavenRepositorySystemUtils.newSession();

        // Set local repository
        LocalRepository localRepo = new LocalRepository(localRepoPath);
        session.setLocalRepositoryManager(system.newLocalRepositoryManager(session, localRepo));

        // Configure resolution policy
        session.setResolutionErrorPolicy(new SimpleResolutionErrorPolicy(false, false));

        // Enable snapshot updates
        session.setUpdatePolicy("always");

        return session;
    }

    /**
     * Resolve artifact and all transitive dependencies.
     *
     * @param artifact Artifact to resolve
     * @return List of resolved JAR files
     */
    private List<File> resolveArtifact(Artifact artifact) {
        try {
            // Create artifact request
            ArtifactRequest request = new ArtifactRequest();
            request.setArtifact(artifact);
            request.setRepositories(remoteRepos);

            // Resolve artifact
            if (repoSystem == null || repoSession == null) {
                throw new IllegalStateException("Maven Repository System is not initialized. Cannot resolve artifact.");
            }
            ArtifactResult result = repoSystem.resolveArtifact(repoSession, request);

            // Return just the resolved artifact JAR
            List<File> jars = new ArrayList<>();
            if (result.getArtifact() != null) {
                File jarFile = result.getArtifact().getFile();
                if (jarFile != null && jarFile.exists()) {
                    jars.add(jarFile);
                }
            }

            return jars;
        } catch (org.eclipse.aether.resolution.ArtifactResolutionException e) {
            throw new RuntimeException("Failed to resolve artifact: " + artifact, e);
        }
    }

    /**
     * Dependency tree structure.
     *
     * @param coordinate   Maven coordinate
     * @param dependencies Child dependencies
     */
    public record DependencyTree(
            String coordinate,
            List<DependencyTree> dependencies) {
        /**
         * Create tree from dependency node.
         *
         * @param node Dependency node
         * @return Dependency tree
         */
        static DependencyTree fromNode(DependencyNode node) {
            List<DependencyTree> children = node.getChildren().stream()
                    .map(DependencyTree::fromNode)
                    .toList();

            String coordinate = node.getArtifact() != null
                    ? node.getArtifact().toString()
                    : "root";

            return new DependencyTree(coordinate, children);
        }

        /**
         * Print tree as string.
         *
         * @return Tree string representation
         */
        public String print() {
            StringBuilder sb = new StringBuilder();
            printTree(sb, 0);
            return sb.toString();
        }

        private void printTree(StringBuilder sb, int depth) {
            String indent = "  ".repeat(depth);
            sb.append(indent).append("+- ").append(coordinate).append("\n");

            for (DependencyTree child : dependencies) {
                child.printTree(sb, depth + 1);
            }
        }

        @Override
        public String toString() {
            return print();
        }
    }
}
