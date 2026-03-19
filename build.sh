#!/bin/bash
set -e

echo "=== Installing root dependencies ==="
pnpm install

echo "=== Building frontend ==="
cd daguailuzi
pnpm install
pnpm run build

echo "=== Installing server dependencies ==="
cd ../server
pnpm install

echo "=== Build completed ==="
