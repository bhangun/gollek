# Gollek Observability Pack

This folder contains starter assets for adapter-aware inference observability and adaptive session-eviction monitoring.

## Files

- `grafana/dashboards/adapter-metrics-overview.json`
  - Grafana dashboard for `adapter.*` metrics by provider/type, including mode-aware routing-filter panels (`managed` vs `standalone/demo`), a top filtered-providers table, and a top filtered-tenants table.
- `grafana/dashboards/session-eviction-overview.json`
  - Grafana dashboard for `gollek.session.eviction.*` metrics by provider.
- `prometheus/adapter-alert-rules.yml`
  - Prometheus alert rules for adapter metrics and session-eviction pressure/reclaim health.

## Metric names expected in Prometheus

Micrometer exports dotted metric names as underscore-separated series:

- `adapter.request.total` -> `adapter_request_total`
- `adapter.session.acquire.duration` -> `adapter_session_acquire_duration_seconds_*`
- `adapter.init.wait.duration` -> `adapter_init_wait_duration_seconds_*`
- `adapter.apply.duration` -> `adapter_apply_duration_seconds_*`
- `adapter.cache.hit` -> `adapter_cache_hit_total`
- `adapter.cache.miss` -> `adapter_cache_miss_total`
- `inference.routing.adapter.filtered` -> `inference_routing_adapter_filtered_total`
- `gollek.session.eviction.pressure.score` -> `gollek_session_eviction_pressure_score`
- `gollek.session.eviction.idle_timeout.seconds` -> `gollek_session_eviction_idle_timeout_seconds`
- `gollek.session.eviction.reclaimed_total` -> `gollek_session_eviction_reclaimed_total`

These queries assume timer `*_sum` and `*_count` series are present.

## Import steps

1. Import both dashboard JSON files into Grafana.
2. Verify top-right dashboard links navigate between Adapter and Session Eviction views.
3. Add `prometheus/adapter-alert-rules.yml` to your Prometheus rule files.
4. Reload Prometheus and verify rule groups `gollek-adapter-metrics` and `gollek-session-eviction-metrics` are loaded.
5. Validate `adapter_request_total` and `gollek_session_eviction_pressure_score` are visible in Prometheus.
6. Validate `inference_routing_adapter_filtered_total` is visible and tagged with `mode`.

## Optional by distribution mode

Grafana/Prometheus is optional. Recommended defaults:

- `gollek-runtime-standalone`: disabled (desktop/demo/individual usage).
- `gollek-runtime-unified`: enabled for server modes, disabled for `%standalone` and `%demo` profiles.
- `gollek-runtime-cloud`: enabled.

If disabled, adapter metric calls are still safe (`NoopAdapterMetricsRecorder` fallback), and inference behavior is unchanged.

### Environment flags

Use these flags to enable/disable observability without rebuilding:

- `GOLLEK_METRICS_ENABLED` (`true|false`)
- `GOLLEK_PROMETHEUS_ENABLED` (`true|false`)
- `GOLLEK_TRACING_ENABLED` (`true|false`)
- `GOLLEK_DISTRIBUTION_MODE` (optional override for cloud runtime mode tag)

Current defaults:

- `gollek-runtime-standalone`: all `false`
- `gollek-runtime-unified`: metrics/prometheus `true`, tracing `false`
- `gollek-runtime-cloud`: all `true`
- `LiteRT` session pool defaults:
  - standalone: `max-per-tenant=2`, `max-total=8`, `idle-timeout-seconds=300`
  - unified (server): `max-per-tenant=4`, `max-total=16`, `idle-timeout-seconds=300`
  - cloud: `max-per-tenant=4`, `max-total=32`, `idle-timeout-seconds=300`

## Initial threshold guidance

- Cache hit ratio warning: `< 85%` for 15m.
- Session acquire latency warning: `> 250ms` for 10m.
- Adapter apply latency critical: `> 350ms` for 10m.
- Init wait warning: `> 500ms` for 10m.
- Session eviction pressure warning: `pressure.score > 0.85` for 10m.
- Session idle-timeout collapse warning: `idle_timeout_seconds < 20` for 15m.
- Session reclaim stall critical: pressure `> 0.70` with reclaim rate near zero for 10m.
- Adapter unsupported-routing filter warning (managed modes): `> 0.2 rps` for 10m.
- Adapter unsupported-routing filter info (standalone/demo modes): `> 2 rps` for 15m.

Tune thresholds per provider after collecting at least 7 days of baseline traffic.
