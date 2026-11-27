"""Feed a source-published configuration into the unchanged pilot fit body."""
import copy
import json
import os
from pathlib import Path
import sys
import warnings
from unittest.mock import patch


def execute(pilot, seeded_fit, job):
    # Replace only the configuration supplier; never merge a default config.
    with patch.object(pilot.paper, 'default_configuration',
                      side_effect=lambda *a, **k: copy.deepcopy(job['configuration'])):
        return seeded_fit(pilot, job['seed'])(
            job['key'], job['task'], job['model'], updates=job['configuration'],
            recipe=job['recipe'], device=job['device'], boundary=False)


def main(plan_path, index):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_bytes())
    job = plan['jobs'][int(index)]
    folder = plan_path.parent
    root = Path(os.environ['TABULAR_PILOT_ROOT'])
    sys.path[:0] = [str(folder), str(root / 'tools/benchmarks'), str(root / 'scripts')]
    import representative_pilot as pilot
    from repeat_seed_adapter import seeded_fit
    from demo_pilot_identity import scientific_key
    assert job['key'] == scientific_key(job['task'], job['model'], job['seed'], job['configuration'])
    assert job['seed'] in job['required_seeds'] and plan['fit_timeout_seconds'] == 1800
    assert job['published_provenance']['no_default_config_merged']
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = execute(pilot, seeded_fit, job)
    assert result['config'] == job['configuration']
    record = dict(key=job['key'], job=job, result=result,
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])
    with (folder / 'runs' / job['key'] / 'demo-result.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
