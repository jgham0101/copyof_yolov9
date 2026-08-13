
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import utils.e2e_postprocess as e2e
from research.week20.official_v10_postprocess import (
    official_v10_two_stage_postprocess,
)

# Only this evaluation process is monkeypatched.
# Core val_e2e.py and core postprocess source are not modified.
e2e.v10_no_nms_postprocess = official_v10_two_stage_postprocess

argv = sys.argv[1:]

if "--postprocess" not in argv:
    argv += ["--postprocess", "no-nms"]

sys.argv = ["val_e2e.py"] + argv

runpy.run_path(
    str(ROOT / "val_e2e.py"),
    run_name="__main__",
)
