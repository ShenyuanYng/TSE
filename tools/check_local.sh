#!/bin/bash
# Lightweight syntax checks only; no Torch/CUDA imports or training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
git diff --check
git diff --cached --check
"${CHECK_PYTHON:-python3}" - <<'PY'
import ast
from pathlib import Path
import subprocess

files = subprocess.check_output(
    ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
).decode().split('\0')
checked = 0
for name in sorted(set(files)):
    path = Path(name)
    if not path.is_file():
        continue
    if path.suffix == '.py':
        ast.parse(path.read_bytes(), filename=name)
    elif path.suffix in ('.sh', '.sbatch'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    else:
        continue
    checked += 1
print(f'Local checks passed: {checked} Python/shell files; Git whitespace checks passed.')
PY
