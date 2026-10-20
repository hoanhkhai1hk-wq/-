"""Candidate offline RGB single-image CLI (INF-20261020-002).

Runs in a fresh process only:
  .venv-recognition/Scripts/python.exe -B scripts/rgb/predict_candidate.py \
      --image <512x512 RGB> --output-dir <absent directory> [--manifest ...] [--seed ...]

The project root is derived from this file's location, so any cwd works.
Candidates are connected Field pixels only - not verified real field parcels,
and pixel areas are not square meters.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.rgb_candidate import (  # noqa: E402
    DEFAULT_MANIFEST, DEFAULT_SEED, CandidatePredictor, export)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--image', required=True, help='512x512 RGB input image')
    parser.add_argument('--output-dir', required=True,
                        help='output directory; it must not already exist')
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST),
                        help='candidate model manifest JSON')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    predictor = CandidatePredictor(args.manifest, seed=args.seed, project_root=ROOT)
    metadata = export(predictor, Path(args.image), Path(args.output_dir))
    print(metadata['candidate_count'], 'candidates;',
          'outputs in', Path(args.output_dir).resolve())
    return 0


if __name__ == '__main__':
    sys.exit(main())
