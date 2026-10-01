Runtime-level fixes to the pinned image, applied by every `mrun` call (MILES_NO_BASE_PATCHES=1 skips).
- pip-requirements.txt: opentelemetry-api 1.44.0 to match the image's opentelemetry-sdk 1.44.0 (Ray agent crash otherwise).
