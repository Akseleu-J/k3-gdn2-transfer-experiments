#!/usr/bin/env bash
# Runs a workflow step and keeps its output, so a failure can be published as an annotation.
bash --noprofile --norc -eo pipefail "$1" 2>&1 | tee -a "$GITHUB_WORKSPACE/ci.log"
exit "${PIPESTATUS[0]}"
