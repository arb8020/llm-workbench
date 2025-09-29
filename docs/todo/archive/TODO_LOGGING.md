# Logging Integration TODO

- [ ] Update `examples/gsm8k_remote_nnsight/server/server.py` to initialize logging via `shared.logging_config.configure_logging()` at startup.
- [ ] Wire `broker` CLI/SDK entry points to use the shared JSON logging formatter; ensure per-instance job IDs flow through log context.
- [ ] Apply the shared logging setup in `bifrost` client and deployment helpers so remote exec logs emit structured metadata.
- [ ] Document the logging standard (How to enable, expected fields, log shipping plan) in `shared/logging.md` and cross-link from the main README files.
