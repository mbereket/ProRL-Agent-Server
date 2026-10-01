#!/usr/bin/env bash
# MTP A/B in one allocation: arm A with the base patch set, arm B (+27B layouts) with patches/no-mtp added.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
bash "${MR}/bench/run.sh" q35-9b-v1 arms-mtp-a.txt bench/patches
bash "${MR}/bench/run.sh" q35-9b-v1 arms-mtp-b.txt bench/patches:patches/no-mtp
