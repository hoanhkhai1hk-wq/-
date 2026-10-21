"""Keep all new test fixtures/logs inside this task's reports directory."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
task_temp = OUT / 'new-test-fixtures'
task_temp.mkdir(exist_ok=False)
env = dict(os.environ)
env.update({'TMP': str(task_temp), 'TEMP': str(task_temp), 'TMPDIR': str(task_temp),
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'})
commands = [
    ['-m', 'unittest', 'discover', '-s', 'tests/labels', '-p', 'test_candidate.py'],
    ['-m', 'unittest', 'discover', '-s', 'tests/labels', '-p', 'test_new_source_scope.py'],
    ['-m', 'unittest', 'discover', '-s', 'tests/recognition', '-p', 'test_frozen_evaluation.py'],
    [str(OUT / 'test_diagnose.py')],
    [str(OUT / 'test_boundary_semantics.py')],
]
results = []
for i, args in enumerate(commands):
    command = [sys.executable, '-B', *args]
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    log = run.stdout + run.stderr
    with (OUT / f'tests-{i + 1}.log').open('x', encoding='utf-8') as f:
        f.write(log)
    match = re.search(r'Ran (\d+) tests?', log)
    result = {'args': args, 'exit_code': run.returncode, 'tests': int(match.group(1)) if match else 0,
              'log': f'tests-{i + 1}.log'}
    results.append(result)
    print(result, flush=True)
    if run.returncode:
        raise SystemExit('Test failed; stop without changing existing files')
with (OUT / 'tests-summary.json').open('x', encoding='utf-8') as f:
    json.dump({'passed': True, 'total_tests': sum(r['tests'] for r in results), 'results': results,
               'temporary_writes_confined_to_reports': str(task_temp.relative_to(ROOT))}, f, ensure_ascii=False, indent=2)
