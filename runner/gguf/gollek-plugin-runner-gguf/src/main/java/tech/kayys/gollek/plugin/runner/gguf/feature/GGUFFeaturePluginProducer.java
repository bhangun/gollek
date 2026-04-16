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

package tech.kayys.gollek.plugin.runner.gguf.feature;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * CDI producer for GGUF feature plugins.
 *
 * @since 2.1.0
 * 
 * @deprecated Use gollek-plugin-feature-text module instead
 */
@ApplicationScoped
@Deprecated
public class GGUFFeaturePluginProducer {

    // Feature plugins are now in separate module: gollek-plugin-feature-text
}
