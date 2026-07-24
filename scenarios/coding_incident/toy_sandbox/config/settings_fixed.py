"""Fixed version of the runtime settings after the incident."""

import os


FIREWORKS_BASE_URL = os.environ.get(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1",
)


try:
    FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
except KeyError as exc:
    raise RuntimeError(
        "FIREWORKS_API_KEY must be supplied by the secret manager or environment"
    ) from exc


REQUEST_TIMEOUT_SECONDS = 30

