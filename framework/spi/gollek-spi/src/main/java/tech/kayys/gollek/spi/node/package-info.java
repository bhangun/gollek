/**
 * Node-based workflow SPI for the Gollek visual graph engine.
 *
 * <p>This package defines the contracts for a visual, node-based workflow system
 * where each computational unit is a {@link tech.kayys.gollek.spi.node.NodePlugin}
 * with typed I/O {@link tech.kayys.gollek.spi.node.NodePort ports}. The Flutter
 * visual editor serializes {@link tech.kayys.gollek.spi.node.NodeDescriptor descriptors}
 * to render the node palette and validate connections.</p>
 *
 * <h2>Key Types</h2>
 * <ul>
 *   <li>{@link tech.kayys.gollek.spi.node.NodePlugin} — Core node abstraction (extends {@code GollekPlugin})</li>
 *   <li>{@link tech.kayys.gollek.spi.node.NodePort} — Typed input/output port descriptor</li>
 *   <li>{@link tech.kayys.gollek.spi.node.NodeDescriptor} — Complete visual metadata record</li>
 *   <li>{@link tech.kayys.gollek.spi.node.NodeCategory} — Visual grouping for the palette</li>
 *   <li>{@link tech.kayys.gollek.spi.node.NodeExecutionContext} — Reactive execution context</li>
 *   <li>{@link tech.kayys.gollek.spi.node.NodeRegistry} — Discovery and resolution</li>
 *   <li>InferencePhaseNodeAdapter (in gollek-plugin-core) — Adapter for legacy phase plugins</li>
 * </ul>
 *
 * @since 3.0.0
 */
package tech.kayys.gollek.spi.node;
