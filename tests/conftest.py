import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

BACKTEST_CHARTS_STR = str(PROJECT_ROOT / "backtest_charts")
sys.path[:] = [path for path in sys.path if not str(path).startswith(BACKTEST_CHARTS_STR)]
for module_name, module in list(sys.modules.items()):
    module_file = getattr(module, "__file__", None)
    if module_name == "src_refactor" or module_name.startswith("src_refactor."):
        if module_file is not None and str(module_file).startswith(BACKTEST_CHARTS_STR):
            del sys.modules[module_name]
