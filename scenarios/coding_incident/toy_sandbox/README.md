# Repo Snapshot: API Key Exposure

This directory is a checkout of the repo state at the bad commit in the coding
incident.

## Files

- `config/settings.py` is the bad commit: CodeAgent inlined the production key
  fallback directly into source.
- `config/settings_fixed.py` shows the intended remediation: read from the
  environment/secret manager instead of source.
- `agent_workforce/fireworks_client.py` shows the application path that imports
  the key.
- `tools/secret_scan.py` is a tiny reproducible scanner used to generate the
  audit evidence.

## Reproduce the Finding

From this directory:

```bash
python3 tools/secret_scan.py .
```

Expected result: the scanner flags `config/settings.py` and exits non-zero.

