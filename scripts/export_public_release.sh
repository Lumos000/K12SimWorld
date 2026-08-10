#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 /path/to/clean/K12SimWorld"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESTINATION="$(realpath -m "$1")"

if [[ "$DESTINATION" == "/" || "$DESTINATION" == "$PROJECT_ROOT" ]]; then
  echo "Refusing unsafe destination: $DESTINATION"
  exit 2
fi

mkdir -p "$DESTINATION"
# GitHub can initialize an otherwise-empty repository with an MIT LICENSE.
# Allow only that single file and replace it with the combined upstream/project
# notice exported below. Any other existing content still requires manual review.
if find "$DESTINATION" -mindepth 1 -maxdepth 1 ! -name .git ! -name LICENSE -print -quit | grep -q .; then
  echo "Destination contains files other than .git and LICENSE: $DESTINATION"
  exit 2
fi
if [[ -f "$DESTINATION/LICENSE" ]]; then
  echo "Existing GitHub LICENSE will be replaced by the combined public LICENSE."
fi

copy_path() {
  local relative="$1"
  local source="$PROJECT_ROOT/$relative"
  if [[ ! -e "$source" ]]; then
    echo "Missing release path: $relative"
    exit 2
  fi
  mkdir -p "$DESTINATION/$(dirname "$relative")"
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.mp4' \
    --exclude '*.webm' \
    "$source" "$DESTINATION/$(dirname "$relative")/"
}

release_paths=(
  .env.template
  .github
  .gitignore
  CONTRIBUTING.md
  LICENSE
  NOTICE
  README.md
  SECURITY.md
  assets/threejs
  docs/API_CONFIGURATION_CN.md
  docs/REPOSITORY_RELEASE_CN.md
  k12simworld
  k12simworld_tests
  package-lock.json
  package.json
  requirements-k12.txt
  run_k12_screening.py
  run_k12simworld.py
  scripts/export_public_release.sh
  src/api_config.py
  src/cannon.min.js
  src/canvas_html_renderer.py
  src/domain_canvas_renderer.py
  src/llm_client.py
  src/manim_renderer.py
  src/p5js_renderer.py
  src/recording.js
  src/three.min.js
  src/threejs_renderer.py
  src/video_normalizer.py
  tests/test_api_config.py
  tests/test_llm_client_security.py
)

for path in "${release_paths[@]}"; do
  copy_path "$path"
done

echo "Public release tree exported to: $DESTINATION"
echo "Review it before git add/commit/push. No Git command was executed."
