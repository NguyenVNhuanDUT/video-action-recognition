"""Compatibility entry point for the V1 dashboard and comparison script."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from violence_project.web.server import DEFAULT_CK, main, predictor, replay

if __name__ == "__main__":
    main()
