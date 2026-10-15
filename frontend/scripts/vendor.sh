#!/usr/bin/env bash
# Re-copy vendored browser libraries into public/static/vendor/ (offline-safe UI).
# Deps are devDependencies: bun add -d marked dompurify @highlightjs/cdn-assets
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p public/static/vendor
cp node_modules/marked/lib/marked.umd.js                           public/static/vendor/marked.umd.js
cp node_modules/dompurify/dist/purify.min.js                       public/static/vendor/purify.min.js
cp node_modules/@highlightjs/cdn-assets/highlight.min.js           public/static/vendor/highlight.min.js
cp node_modules/@highlightjs/cdn-assets/styles/github-dark.min.css public/static/vendor/highlight.css

echo "vendor/ updated ✓ (marked · DOMPurify · highlight.js)"
