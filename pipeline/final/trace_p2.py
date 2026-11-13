"""trace.py with the corrected simulator (tank energy balance in memory + airflow-weighted return air); writes <scratch>/trace_p2.
python3 -B pipeline/launch/launch.py trace_p2.py <scratch dir> <model name> <scenario dir>"""
import os, sys
from pathlib import Path
os.environ["SB_RETURN_AIR"] = "flow"; os.environ["SB_TANK"] = "balance"
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import run_c                                   # noqa: E402  (reads sys.argv[1] = scratch dir)
for p in (str(run_c.run_x.ROOT / "packages"), str(run_c.run_x.DATA)):
    if p not in sys.path: sys.path.insert(0, p)
run_c._tank_balance()                          # installs the fast simulator and patches the boiler in this process
src = (here / "trace.py").read_text(encoding="utf-8").replace('out = SCR / "trace"', 'out = SCR / "trace_p2"')
assert 'SCR / "trace_p2"' in src
exec(compile(src, str(here / "trace.py"), "exec"))
