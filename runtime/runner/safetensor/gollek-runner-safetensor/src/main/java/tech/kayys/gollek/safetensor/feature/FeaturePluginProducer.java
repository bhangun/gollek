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

package tech.kayys.gollek.plugin.runner.safetensor.feature;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.stream.StreamSupport;

/**
 * CDI producer for Safetensor feature plugins.
 *
 * <p>Discovers and produces feature plugin instances dynamically using CDI.</p>
 */
@ApplicationScoped
public class FeaturePluginProducer {

    private static final Logger LOG = Logger.getLogger(FeaturePluginProducer.class);

    @Inject
    Instance<SafetensorFeaturePlugin> featurePluginInstances;

    /**
     * Produce all discovered feature plugins.
     *
     * @return List of discovered feature plugin instances
     */
    @Produces
    @Singleton
    public List<SafetensorFeaturePlugin> produceFeaturePlugins() {
        if (featurePluginInstances == null || featurePluginInstances.isUnsatisfied()) {
            LOG.warn("No feature plugins discovered via CDI");
            return List.of();
        }

        List<SafetensorFeaturePlugin> plugins = StreamSupport
            .stream(featurePluginInstances.spliterator(), false)
            .peek(plugin -> LOG.infof("Discovered feature plugin: %s (%s)", plugin.name(), plugin.id()))
            .toList();

        LOG.infof("Discovered %d feature plugins", plugins.size());
        return plugins;
    }
}
