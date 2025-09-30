# ✅ This is the Correct GSM8K NNsight Remote Example

## What Happened

There were **two** nnsight examples:

1. **`examples/dev/deprecated/gsm8k_nnsight_remote/`** - Sept 12-14, 2025
   - Original attempt with `server_singlepass.py` and `server_composition.py`
   - Had nnsight v0.5.0 compatibility issues
   - Monolithic server files (~500 lines each)
   - Deprecated and moved to `examples/dev/deprecated/`

2. **`examples/dev/gsm8k_remote_nnsight/`** (THIS ONE) - Sept 14, 2025+
   - Better structured with `server/` directory
   - Separate modules: `server.py`, `activation_capture.py`, `config.py`
   - Proper intervention configuration API
   - More mature and actively used

## Key Differences

### Deprecated Version (DON'T USE)
```
gsm8k_nnsight_remote/
├── server_singlepass.py     # 500+ lines, broken
├── server_composition.py    # 200+ lines, broken
├── deploy_and_smoke_test.py
└── README.md
```

### Correct Version (USE THIS)
```
gsm8k_remote_nnsight/
├── server/
│   ├── server.py              # Main FastAPI server
│   ├── activation_capture.py  # Tensor saving logic
│   ├── config.py              # Intervention config
│   ├── test_client.py         # Testing utility
│   └── smoke.py               # Smoke tests
├── deploy.py                  # Deployment script
├── deploy_and_collect.py      # Full evaluation
├── HANDOFF_DEBUG.md           # Debugging guide
└── README.md
```

## Timeline

- **Sept 12**: First attempt created at `examples/gsm8k_nnsight_remote/`
- **Sept 14**: Refactored into better structure at `examples/gsm8k_remote_nnsight/`
- **Sept 29**: First attempt moved to `examples/dev/deprecated/gsm8k_nnsight_remote/`
- **Sept 29 (later)**: Mistakenly "revived" the deprecated version
- **Sept 29 (now)**: Corrected - using the proper `examples/dev/gsm8k_remote_nnsight/`

## How to Use

```bash
# Deploy server
python examples/dev/gsm8k_remote_nnsight/deploy.py \
  --name nnsight-test \
  --port 8000

# Run evaluation with activation collection
python examples/dev/gsm8k_remote_nnsight/deploy_and_collect.py \
  --samples 3
```

## References

- Main README: `examples/dev/gsm8k_remote_nnsight/README.md`
- Debug guide: `examples/dev/gsm8k_remote_nnsight/HANDOFF_DEBUG.md`
- Debugging context: `examples/dev/gsm8k_remote_nnsight/DEBUGGING_CONTEXT.md` (needs update)

## Status

The DEBUGGING_CONTEXT.md file currently references the old broken code and needs to be updated to reference this correct implementation.