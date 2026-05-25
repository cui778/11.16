from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
FIG_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = FIG_ROOT / "outputs"
INPUT_DIR = SCRIPT_ROOT / "input_1"
REPORTS_DIR = SCRIPT_ROOT / "outputs" / "reports"
CURRENT_CLEAN_SUBDIR = "time_gated_node_sensor_v2_rebuild"
CURRENT_CLEAN_PARQUET = SCRIPT_ROOT / "training_data_new" / CURRENT_CLEAN_SUBDIR / "node_timeseries_with_residuals.parquet"
CURRENT_DEFECT_CSV = INPUT_DIR / "defect_matrix_diverse.csv"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _pick_cjk_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return font_manager.FontProperties(fname=str(path)).get_name()
            except Exception:
                continue
    return "DejaVu Sans"


def setup_style() -> None:
    cjk_font = _pick_cjk_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [cjk_font, "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_list(path: Path, keys: tuple[str, ...]) -> list[str]:
    data = load_json(path)
    if isinstance(data, list):
        return [str(x) for x in data]
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]
    raise KeyError(f"{path} 中未找到任何候选键: {keys}")


def save_fig(fig, name: str) -> Path:
    out = ensure_output_dir() / name
    fig.savefig(out, bbox_inches="tight")
    return out


def load_metrics_json(name: str) -> dict:
    return load_json(REPORTS_DIR / name)


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"未找到可用文件，已尝试:\n{joined}")
