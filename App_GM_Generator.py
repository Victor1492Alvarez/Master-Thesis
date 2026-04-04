from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


APP_TITLE = "Gaussian Model Generator"
APP_SUBTITLE = "PtMeOH Gaussian surrogate modeling from Aspen Plus Excel data"
RANDOM_STATE = 42

# ------------------------------------------------------------------
# CENTRAL SETTINGS
# Ajusta aquí la relación de aspecto de los gráficos y las fuentes.
# ------------------------------------------------------------------
PLOT_SETTINGS = {
    "prediction_figsize": (8.8, 8.8),
    "percent_error_figsize": (12.0, 5.0),
    "cv_figsize": (8.8, 5.0),
    "external_comparison_figsize": (8.8, 5.0),
    "external_error_figsize": (12.0, 5.0),
    "title_fontsize": 10,
    "axis_label_fontsize": 9,
    "tick_x_fontsize": 5.0,
    "tick_y_fontsize": 10,
    "legend_fontsize": 10,
}

PDF_SETTINGS = {
    "pagesize": landscape(A4),
    "left_margin_cm": 1.2,
    "right_margin_cm": 1.2,
    "top_margin_cm": 1.1,
    "bottom_margin_cm": 1.1,
    "section_gap_cm": 1.0,
    "small_gap_cm": 0.18,
    "table_chart_gap_cm": 3.0,
    "training_cv_chart_height_cm": 9.0,
    "training_prediction_chart_height_cm": 7.9,
    "training_error_chart_height_cm": 5.8,
    "consolidated_main_chart_height_cm": 8.8,
    "consolidated_error_chart_height_cm": 6.0,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


class GaussianSurrogate1D:
    def __init__(self, kernel_name: str = "Matern 2.5", use_white_kernel: bool = False, alpha: float = 1e-8):
        self.kernel_name = kernel_name
        self.use_white_kernel = use_white_kernel
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.gpr = GaussianProcessRegressor(
            kernel=self._build_kernel(),
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=RANDOM_STATE,
        )
        self.is_fitted = False

    def _build_kernel(self):
        if self.kernel_name == "RBF":
            base = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-3, 1e3),
            )
        else:
            base = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
        if self.use_white_kernel:
            return base + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        return base

    @staticmethod
    def _ensure_2d(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X, y):
        X_arr = self._ensure_2d(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        X_scaled = self.scaler.fit_transform(X_arr)
        self.gpr.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X, return_std: bool = False):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_arr = self._ensure_2d(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.gpr.predict(X_scaled, return_std=return_std)

    @property
    def fitted_kernel(self) -> str:
        return str(self.gpr.kernel_)

    def export_bundle(self, model_name: str, input_column: str, output_column: str) -> Dict:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before export.")
        return {
            "model_name": model_name,
            "input_column": input_column,
            "output_column": output_column,
            "kernel_name": self.kernel_name,
            "use_white_kernel": self.use_white_kernel,
            "alpha": self.alpha,
            "fitted_kernel": self.fitted_kernel,
            "created_at": datetime.now().isoformat(),
            "scaler": self.scaler,
            "gpr": self.gpr,
        }


def init_state() -> None:
    defaults = {
        "app_initialized": False,
        "current_step": 1,
        "module_status": {
            "analyzer": "ready",
            "cleaning": "locked",
            "training": "locked",
            "testing": "locked",
        },
        "logs": {
            "analyzer": [],
            "cleaning": [],
            "training": [],
            "testing": [],
        },
        "uploaded_file_name": None,
        "raw_df": None,
        "clean_df": None,
        "train_val_df": None,
        "external_test_df": None,
        "input_column": None,
        "output_column": None,
        "status_column": None,
        "detected_input_candidates": [],
        "run_summary": {},
        "analyzer_done": False,
        "cleaning_done": False,
        "training_done": False,
        "testing_done": False,
        "model_name": "PtMeOH_GP_Model",
        "cv_fold_results": [],
        "cv_summary": None,
        "cv_comparison": None,
        "temporary_parameter_log": [],
        "trained_model": None,
        "trained_model_bundle": None,
        "trained_model_metadata": None,
        "external_test_results": None,
        "suspicious_points": None,
        "diagnostics_text": [],
        "final_approved": False,
        "artifacts": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workflow() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.rerun()


def add_log(module: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"][module].append(f"[{timestamp}] {message}")


def set_module_status(module: str, status: str) -> None:
    st.session_state["module_status"][module] = status


def parse_yes_no(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    normalized = str(text).strip().lower()
    if normalized in {"y", "yes"}:
        return True
    if normalized in {"n", "no"}:
        return False
    return None


def status_badge_html(status: str) -> str:
    colors_map = {
        "locked": ("#b5b5b5", "#666666"),
        "ready": ("#efefef", "#222222"),
        "running": ("#dddddd", "#111111"),
        "completed": ("#111111", "#ffffff"),
        "error": ("#222222", "#ffffff"),
    }
    bg, fg = colors_map.get(status, ("#efefef", "#222222"))
    return (
        f"<span style='display:inline-block;padding:0.35rem 0.7rem;"
        f"border:1px solid #111;border-radius:999px;background:{bg};color:{fg};"
        f"font-size:0.85rem;font-weight:700;letter-spacing:0.02em;'>"
        f"{status.upper()}</span>"
    )


def render_log(module: str, key: str) -> None:
    log_text = "\n".join(st.session_state["logs"][module][-300:])
    st.text_area("Scrolling log panel", value=log_text, height=240, disabled=True, key=key)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .app-title {font-size: 2rem; font-weight: 800; letter-spacing: 0.02em; margin-bottom: 0.15rem;}
        .app-subtitle {font-size: 1rem; color: #444; margin-bottom: 1rem;}
        .module-card {
            border: 1px solid #111;
            border-radius: 0.8rem;
            padding: 1rem 1rem 0.75rem 1rem;
            background: #fff;
            margin-bottom: 0.8rem;
        }
        .locked-card {
            border: 1px dashed #777;
            border-radius: 0.8rem;
            padding: 1rem;
            background: #fafafa;
            color: #444;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            color: #444;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    completed = sum(int(st.session_state[f"{name}_done"]) for name in ["analyzer", "cleaning", "training", "testing"])
    progress = completed / 4.0
    st.progress(progress, text=f"Workflow progress: {int(progress * 100)}%")


def render_module_header(module_key: str, title: str, subtitle: str, notes_key: str) -> None:
    st.markdown("<div class='module-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 2.8])
    with c1:
        st.markdown(status_badge_html(st.session_state["module_status"][module_key]), unsafe_allow_html=True)
    with c2:
        st.text_area(
            "User interaction / command area",
            key=notes_key,
            height=90,
            placeholder="Type short module notes, instructions, or operator comments...",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def show_locked_message(text: str) -> None:
    st.markdown(f"<div class='locked-card'>{text}</div>", unsafe_allow_html=True)


def detect_input_candidates(columns: List[str]) -> List[str]:
    patterns = ["sh2", "h2", "mixed", "total mole flow", "mol/hr"]
    scored = []
    for col in columns:
        text = str(col).strip().lower()
        score = sum(1 for p in patterns if p in text)
        if score > 0:
            scored.append((col, score))
    scored.sort(key=lambda x: (-x[1], str(x[0]).lower()))
    return [name for name, _ in scored]


def find_status_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).strip().lower() == "status":
            return col
    return None


def normalize_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def choose_excel_engine(filename: str) -> Optional[str]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    return None


def read_uploaded_excel(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No Aspen Excel file has been uploaded.")
    engine = choose_excel_engine(uploaded_file.name)
    if engine is None:
        raise ValueError("Unsupported file type. Please upload .xlsx, .xlsm, or .xls.")
    uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine=engine)
    except ImportError as exc:
        if engine == "xlrd":
            raise ImportError("Reading .xls files requires xlrd. Install it with: pip install xlrd") from exc
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not read the Excel file: {exc}") from exc


def to_excel_bytes(sheet_map: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def joblib_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def safe_percent_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    return np.abs(y_true - y_pred) / denom * 100.0


def summarize_runs(df: pd.DataFrame, input_col: str, output_col: Optional[str], status_col: Optional[str] = None) -> Dict[str, float]:
    if input_col not in df.columns:
        return {
            "total_runs": 0,
            "successful_runs": 0,
            "erroneous_runs": 0,
            "ok_percentage": 0.0,
            "err_percentage": 0.0,
        }

    input_numeric = pd.to_numeric(df[input_col], errors="coerce")
    total_runs = int(input_numeric.notna().sum())

    status_mask = pd.Series(True, index=df.index)
    if status_col and status_col in df.columns:
        status_mask = normalize_status(df[status_col]).eq("ok")

    if output_col and output_col in df.columns:
        output_numeric = pd.to_numeric(df[output_col], errors="coerce")
        successful_mask = input_numeric.notna() & output_numeric.notna() & status_mask
        successful = int(successful_mask.sum())
    else:
        successful = 0

    erroneous = max(total_runs - successful, 0)
    ok_pct = (successful / total_runs * 100.0) if total_runs else 0.0
    err_pct = (erroneous / total_runs * 100.0) if total_runs else 0.0

    return {
        "total_runs": total_runs,
        "successful_runs": successful,
        "erroneous_runs": erroneous,
        "ok_percentage": round(ok_pct, 2),
        "err_percentage": round(err_pct, 2),
    }


def representative_split(
    df: pd.DataFrame,
    input_col: str,
    test_frac: float = 0.2,
    bins: int = 10,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 6:
        raise ValueError("At least 6 valid rows are recommended to create a representative external test split.")

    work = df.sort_values(input_col).reset_index(drop=True).copy()
    total_n = len(work)
    target_test_n = max(1, int(round(total_n * test_frac)))
    target_test_n = min(target_test_n, total_n - 5) if total_n > 5 else 1
    target_test_n = max(target_test_n, 1)

    ranked = work[input_col].rank(method="first")
    q = min(bins, max(2, min(10, total_n // 2)))
    work["_bin"] = pd.qcut(ranked, q=q, labels=False, duplicates="drop")

    rng = np.random.default_rng(random_state)
    chosen_indices: List[int] = []

    for _, group in work.groupby("_bin"):
        n_group = len(group)
        n_take = max(1, int(round(n_group * target_test_n / total_n)))
        n_take = min(n_take, n_group)
        sampled = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        chosen_indices.extend(sampled.tolist())

    chosen_indices = sorted(set(chosen_indices))

    if len(chosen_indices) > target_test_n:
        chosen_indices = rng.choice(np.array(chosen_indices), size=target_test_n, replace=False).tolist()
    elif len(chosen_indices) < target_test_n:
        remaining = sorted(set(work.index) - set(chosen_indices))
        needed = target_test_n - len(chosen_indices)
        extra = rng.choice(np.array(remaining), size=needed, replace=False).tolist()
        chosen_indices.extend(extra)

    chosen_indices = sorted(set(chosen_indices))
    test_df = work.loc[chosen_indices].drop(columns="_bin").sort_values(input_col).reset_index(drop=True)
    train_df = work.drop(index=chosen_indices).drop(columns="_bin").sort_values(input_col).reset_index(drop=True)

    if len(train_df) < 5:
        raise ValueError("Training-validation set is too small for 5-fold cross-validation after the split.")
    return train_df, test_df


def _pdf_is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _pdf_format_value(value, decimals: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _estimate_pdf_col_widths(df: pd.DataFrame, available_width: float, max_rows: int = 20, decimals: int = 4) -> List[float]:
    preview = df.head(max_rows).copy()
    raw_widths = []
    for col in preview.columns:
        samples = [str(col)]
        samples.extend(_pdf_format_value(v, decimals) for v in preview[col].tolist())
        max_len = max(len(x) for x in samples) if samples else 8
        raw_widths.append(min(max(max_len, 7), 24))
    total = sum(raw_widths) if raw_widths else 1
    return [available_width * (w / total) for w in raw_widths]


def simple_table_from_df(
    df: pd.DataFrame,
    max_rows: int = 20,
    available_width: Optional[float] = None,
    decimals: int = 4,
    font_size: float = 6.8,
    header_font_size: float = 7.0,
    justify_cols: Optional[List[str]] = None,
) -> Table:
    preview = df.head(max_rows).copy().fillna("")
    available_width = available_width or (16.8 * cm)
    justify_cols = justify_cols or []

    header_style = ParagraphStyle(
        "pdf_header_style",
        fontName="Helvetica-Bold",
        fontSize=header_font_size,
        leading=header_font_size + 1.1,
        alignment=TA_CENTER,
        wordWrap="CJK",
        textColor=colors.black,
    )
    num_style = ParagraphStyle(
        "pdf_num_style",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.0,
        alignment=TA_RIGHT,
        wordWrap="CJK",
    )
    text_style = ParagraphStyle(
        "pdf_text_style",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.0,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    justify_style = ParagraphStyle(
        "pdf_justify_style",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.0,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
    )

    col_widths = _estimate_pdf_col_widths(preview, available_width=available_width, max_rows=max_rows, decimals=decimals)

    rows = [[Paragraph(str(col), header_style) for col in preview.columns]]
    for _, row in preview.iterrows():
        row_cells = []
        for col, value in row.items():
            text = _pdf_format_value(value, decimals=decimals).replace("\n", "<br/>")
            if _pdf_is_number(value):
                row_cells.append(Paragraph(text, num_style))
            else:
                style = justify_style if str(col) in justify_cols else text_style
                row_cells.append(Paragraph(text, style))
        rows.append(row_cells)

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f6")]),
            ]
        )
    )
    return table


def create_prediction_plot(
    df_plot: pd.DataFrame,
    input_col: str,
    output_col: str,
    title: str,
    uncertainty_col: Optional[str] = None,
) -> bytes:
    ordered = df_plot.sort_values(input_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["prediction_figsize"])

    ax.scatter(ordered[input_col], ordered[output_col], color="black", s=28, label="Aspen")
    ax.plot(ordered[input_col], ordered["Prediction"], color="black", linewidth=1.8, label="Gaussian model")

    if uncertainty_col and uncertainty_col in ordered.columns:
        lower = ordered["Prediction"] - 1.96 * ordered[uncertainty_col]
        upper = ordered["Prediction"] + 1.96 * ordered[uncertainty_col]
        ax.fill_between(ordered[input_col], lower, upper, color="0.82", alpha=1.0, label="95% interval")

    ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"])
    ax.set_xlabel(input_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.set_ylabel(output_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.tick_params(axis="x", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, fontsize=PLOT_SETTINGS["legend_fontsize"])
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def create_error_plot(df_plot: pd.DataFrame, input_col: str, title: str) -> bytes:
    ordered = df_plot.sort_values(input_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["percent_error_figsize"])

    x = np.arange(len(ordered))
    ax.bar(x, ordered["Percent Error"], color="0.25", edgecolor="black", linewidth=0.3)

    ax.set_title(title, fontsize=PLOT_SETTINGS["title_fontsize"])
    ax.set_xlabel(input_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.set_ylabel("Percent Error [%]", fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.4f}" for v in ordered[input_col].tolist()], rotation=90)
    ax.tick_params(axis="x", labelsize=PLOT_SETTINGS["tick_x_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.grid(True, axis="y", color="0.85", linewidth=0.8)
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def create_cv_metrics_plot(metrics_df: pd.DataFrame) -> bytes:
    fig, ax1 = plt.subplots(figsize=PLOT_SETTINGS["cv_figsize"])
    x = np.arange(len(metrics_df))
    width = 0.34

    ax1.bar(x - width / 2, metrics_df["RMSE"], width=0.20, color="0.20", label="RMSE")
    ax1.bar(x + width / 2, metrics_df["MAE"], width=0.20, color="0.65", label="MAE")
    ax1.set_xlabel("Fold", fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax1.set_ylabel("Error magnitude", fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df["Fold"].astype(str))
    ax1.tick_params(axis="x", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax1.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax1.grid(True, axis="y", color="0.88", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, metrics_df["R2"], color="black", linewidth=1.8, marker="o", label="R²")
    ax2.set_ylabel("R²", fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax2.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", fontsize=PLOT_SETTINGS["legend_fontsize"])
    ax1.set_title("5-Fold Cross-Validation Metrics", fontsize=PLOT_SETTINGS["title_fontsize"])
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def create_external_comparison_plot(df_plot: pd.DataFrame, input_col: str, output_col: str) -> bytes:
    ordered = df_plot.sort_values(input_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["external_comparison_figsize"])

    ax.scatter(ordered[input_col], ordered[output_col], color="black", s=20, label="Aspen points")
    ax.plot(ordered[input_col], ordered["Prediction"], color="black", linewidth=2.0, label="Gaussian model")
    ax.fill_between(
        ordered[input_col],
        ordered["Prediction"] - 1.96 * ordered["Predictive Std"],
        ordered["Prediction"] + 1.96 * ordered["Predictive Std"],
        color="0.85",
        label="95% interval",
    )
    ax.set_title("External Test: Gaussian Model vs Aspen Data", fontsize=PLOT_SETTINGS["title_fontsize"])
    ax.set_xlabel(input_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.set_ylabel(output_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.tick_params(axis="x", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, fontsize=PLOT_SETTINGS["legend_fontsize"])
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def create_external_error_plot(df_plot: pd.DataFrame, input_col: str) -> bytes:
    ordered = df_plot.sort_values(input_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["external_error_figsize"])

    x = np.arange(len(ordered))
    ax.bar(x, ordered["Percent Error"], color="0.4", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.4f}" for v in ordered[input_col].tolist()], rotation=90)
    ax.tick_params(axis="x", labelsize=PLOT_SETTINGS["tick_x_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_SETTINGS["tick_y_fontsize"])
    ax.set_title("External Test: Percent Error by Run", fontsize=PLOT_SETTINGS["title_fontsize"])
    ax.set_xlabel(input_col, fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.set_ylabel("Percent Error [%]", fontsize=PLOT_SETTINGS["axis_label_fontsize"])
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def perform_cv(
    df: pd.DataFrame,
    input_col: str,
    output_col: str,
    kernel_name: str,
    use_white_kernel: bool,
    alpha_value: float,
    enable_logs: bool = True,
) -> Dict:
    if len(df) < 5:
        raise ValueError("At least 5 rows are required in the training-validation dataset for 5-fold CV.")

    X = df[[input_col]].astype(float).to_numpy()
    y = df[output_col].astype(float).to_numpy()

    if np.unique(X).size < 2:
        raise ValueError("The selected input variable does not have enough variation to train a Gaussian model.")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    parameter_log = []
    fold_tables = {}
    fold_prediction_plots = {}
    fold_error_plots = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        if enable_logs:
            add_log("training", f"Fold {fold_idx}/5 started using kernel {kernel_name}.")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GaussianSurrogate1D(
            kernel_name=kernel_name,
            use_white_kernel=use_white_kernel,
            alpha=alpha_value,
        ).fit(X_train, y_train)

        y_pred, y_std = model.predict(X_val, return_std=True)
        metrics = compute_metrics(y_val, y_pred)
        percent_error = safe_percent_error(y_val, y_pred)

        fold_df = pd.DataFrame(
            {
                input_col: X_val.flatten(),
                output_col: y_val,
                "Prediction": y_pred,
                "Predictive Std": y_std,
                "Absolute Error": np.abs(y_val - y_pred),
                "Percent Error": percent_error,
            }
        ).sort_values(input_col).reset_index(drop=True)

        parameter_log.append(
            {
                "Fold": fold_idx,
                "Kernel Name": kernel_name,
                "Fitted Kernel": model.fitted_kernel,
                "Alpha": alpha_value,
                "Use WhiteKernel": use_white_kernel,
                "Train Size": len(train_idx),
                "Validation Size": len(val_idx),
                "Train X Min": float(np.min(X_train)),
                "Train X Max": float(np.max(X_train)),
                "Validation X Min": float(np.min(X_val)),
                "Validation X Max": float(np.max(X_val)),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
            }
        )

        fold_results.append({"Fold": fold_idx, **metrics})
        fold_name = f"Fold_{fold_idx}"
        fold_tables[fold_name] = fold_df
        fold_prediction_plots[f"{fold_name}_prediction.png"] = create_prediction_plot(
            fold_df,
            input_col,
            output_col,
            f"Fold {fold_idx}: Aspen vs Gaussian Prediction",
            uncertainty_col="Predictive Std",
        )
        fold_error_plots[f"{fold_name}_error.png"] = create_error_plot(
            fold_df,
            input_col,
            f"Fold {fold_idx}: Percent Error",
        )

        if enable_logs:
            add_log(
                "training",
                f"Fold {fold_idx}/5 completed. RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}.",
            )

    metrics_df = pd.DataFrame(fold_results)
    summary_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R2"],
            "Mean": [metrics_df["RMSE"].mean(), metrics_df["MAE"].mean(), metrics_df["R2"].mean()],
            "Std": [metrics_df["RMSE"].std(ddof=1), metrics_df["MAE"].std(ddof=1), metrics_df["R2"].std(ddof=1)],
        }
    )

    return {
        "kernel_name": kernel_name,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "parameter_log": parameter_log,
        "fold_tables": fold_tables,
        "fold_prediction_plots": fold_prediction_plots,
        "fold_error_plots": fold_error_plots,
        "cv_metrics_plot": create_cv_metrics_plot(metrics_df),
    }


def select_best_cv_result(result_a: Dict, result_b: Dict) -> Dict:
    rmse_a = float(result_a["summary_df"].loc[result_a["summary_df"]["Metric"] == "RMSE", "Mean"].iloc[0])
    rmse_b = float(result_b["summary_df"].loc[result_b["summary_df"]["Metric"] == "RMSE", "Mean"].iloc[0])
    if rmse_a < rmse_b:
        return result_a
    if rmse_b < rmse_a:
        return result_b

    mae_a = float(result_a["summary_df"].loc[result_a["summary_df"]["Metric"] == "MAE", "Mean"].iloc[0])
    mae_b = float(result_b["summary_df"].loc[result_b["summary_df"]["Metric"] == "MAE", "Mean"].iloc[0])
    return result_a if mae_a <= mae_b else result_b


def build_python_wrapper(bundle_filename: str) -> str:
    return f'''"""Reusable packaged Gaussian surrogate model."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

_BUNDLE = joblib.load(Path(__file__).with_name({bundle_filename!r}))
_MODEL_NAME = _BUNDLE["model_name"]
_INPUT_COLUMN = _BUNDLE["input_column"]
_OUTPUT_COLUMN = _BUNDLE["output_column"]
_SCALER = _BUNDLE["scaler"]
_GPR = _BUNDLE["gpr"]

def _ensure_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def predict(h2_flow):
    arr = _ensure_2d(h2_flow)
    arr_scaled = _SCALER.transform(arr)
    pred, std = _GPR.predict(arr_scaled, return_std=True)

    if arr.shape[0] == 1:
        return {{
            "model_name": _MODEL_NAME,
            "input_column": _INPUT_COLUMN,
            "output_column": _OUTPUT_COLUMN,
            "input": float(arr.flatten()[0]),
            "prediction": float(pred[0]),
            "predictive_std": float(std[0]),
        }}

    return pd.DataFrame({{
        _INPUT_COLUMN: arr.flatten(),
        "Prediction": pred,
        "Predictive Std": std,
        "Model Name": _MODEL_NAME,
        "Output Column": _OUTPUT_COLUMN,
    }})
'''


def build_text_summary(
    model_name: str,
    input_col: str,
    output_col: str,
    cv_summary: pd.DataFrame,
    external_metrics: Optional[Dict[str, float]] = None,
) -> str:
    lines = [
        "Gaussian Model Generator - Model Summary",
        f"Model name: {model_name}",
        f"Input column: {input_col}",
        f"Output column: {output_col}",
        "",
        "Cross-validation summary:",
        cv_summary.to_string(index=False),
    ]
    if external_metrics is not None:
        lines.extend(
            [
                "",
                "External test summary:",
                f"RMSE: {external_metrics['RMSE']:.6f}",
                f"MAE: {external_metrics['MAE']:.6f}",
                f"R2: {external_metrics['R2']:.6f}",
            ]
        )
    return "\n".join(lines)


def _pdf_doc() -> SimpleDocTemplate:
    return SimpleDocTemplate(
        io.BytesIO(),
        pagesize=PDF_SETTINGS["pagesize"],
        rightMargin=PDF_SETTINGS["right_margin_cm"] * cm,
        leftMargin=PDF_SETTINGS["left_margin_cm"] * cm,
        topMargin=PDF_SETTINGS["top_margin_cm"] * cm,
        bottomMargin=PDF_SETTINGS["bottom_margin_cm"] * cm,
    )


def build_training_pdf(
    model_name: str,
    input_col: str,
    output_col: str,
    chosen_cv: Dict,
    comparison_df: Optional[pd.DataFrame],
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=PDF_SETTINGS["pagesize"],
        rightMargin=PDF_SETTINGS["right_margin_cm"] * cm,
        leftMargin=PDF_SETTINGS["left_margin_cm"] * cm,
        topMargin=PDF_SETTINGS["top_margin_cm"] * cm,
        bottomMargin=PDF_SETTINGS["bottom_margin_cm"] * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Meta", fontName="Helvetica", fontSize=8.8, leading=10.8, alignment=TA_LEFT))
    story = []
    page_width = doc.width

    story.append(Paragraph("Training & Validation Report", styles["Title"]))
    story.append(Spacer(1, 0.30 * cm))
    story.append(Paragraph(f"Model name: {model_name}", styles["Meta"]))
    story.append(Paragraph(f"Input variable: {input_col}", styles["Meta"]))
    story.append(Paragraph(f"Output variable: {output_col}", styles["Meta"]))
    story.append(Paragraph(f"Selected kernel: {chosen_cv['kernel_name']}", styles["Meta"]))
    story.append(Spacer(1, 0.25 * cm))

    if comparison_df is not None and not comparison_df.empty:
        story.append(Paragraph("Kernel comparison", styles["Heading2"]))
        story.append(
            simple_table_from_df(
                comparison_df,
                max_rows=10,
                available_width=page_width,
                decimals=4,
                font_size=6.6,
                header_font_size=6.9,
            )
        )
        story.append(Spacer(1, PDF_SETTINGS["section_gap_cm"] * cm))

    story.append(Paragraph("Cross-validation summary and metrics chart", styles["Heading2"]))
    story.append(Spacer(1, PDF_SETTINGS["small_gap_cm"] * cm))

    summary_table = simple_table_from_df(
        chosen_cv["summary_df"],
        max_rows=10,
        available_width=8.2 * cm,
        decimals=4,
        font_size=7.1,
        header_font_size=7.3,
    )
    cv_chart = Image(
        io.BytesIO(chosen_cv["cv_metrics_plot"]),
        width=page_width - 9.0 * cm,
        height=PDF_SETTINGS["training_cv_chart_height_cm"] * cm,
    )
    cv_chart.hAlign = "CENTER"

    first_page_layout = Table(
        [[summary_table, cv_chart]],
        colWidths=[8.6 * cm, page_width - 8.6 * cm],
        hAlign="CENTER",
    )
    first_page_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    story.append(first_page_layout)
    story.append(PageBreak())

    for fold_name, df_fold in chosen_cv["fold_tables"].items():
        plot_key = f"{fold_name}_prediction.png"
        error_key = f"{fold_name}_error.png"

        story.append(Paragraph(f"{fold_name.replace('_', ' ')} results", styles["Heading2"]))
        story.append(Spacer(1, 0.12 * cm))
        story.append(
            simple_table_from_df(
                df_fold,
                max_rows=18,
                available_width=page_width,
                decimals=4,
                font_size=6.0,
                header_font_size=6.3,
            )
        )
        story.append(Spacer(1, 0.28 * cm))

        pred_chart = Image(
            io.BytesIO(chosen_cv["fold_prediction_plots"][plot_key]),
            width=page_width,
            height=PDF_SETTINGS["training_prediction_chart_height_cm"] * cm,
        )
        pred_chart.hAlign = "CENTER"
        story.append(pred_chart)
        story.append(PageBreak())

        story.append(Paragraph(f"{fold_name.replace('_', ' ')} - Percent Error", styles["Heading2"]))
        story.append(Spacer(1, 0.22 * cm))
        err_chart = Image(
            io.BytesIO(chosen_cv["fold_error_plots"][error_key]),
            width=page_width,
            height=PDF_SETTINGS["training_error_chart_height_cm"] * cm,
        )
        err_chart.hAlign = "CENTER"
        story.append(err_chart)
        story.append(PageBreak())

    param_df = pd.DataFrame(chosen_cv["parameter_log"])
    story.append(Paragraph("Kernel parameters and temporary parameter log", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        simple_table_from_df(
            param_df,
            max_rows=25,
            available_width=page_width,
            decimals=3,
            font_size=5.8,
            header_font_size=6.1,
            justify_cols=["kernel_parameters", "Fitted Kernel", "kernel_name", "Kernel Name"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_consolidated_pdf(
    model_name: str,
    input_col: str,
    output_col: str,
    cv_summary: pd.DataFrame,
    external_results_df: pd.DataFrame,
    external_metrics: Dict[str, float],
    external_plot_bytes: bytes,
    external_error_plot_bytes: bytes,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=PDF_SETTINGS["pagesize"],
        rightMargin=PDF_SETTINGS["right_margin_cm"] * cm,
        leftMargin=PDF_SETTINGS["left_margin_cm"] * cm,
        topMargin=PDF_SETTINGS["top_margin_cm"] * cm,
        bottomMargin=PDF_SETTINGS["bottom_margin_cm"] * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Meta", fontName="Helvetica", fontSize=8.8, leading=10.8, alignment=TA_LEFT))
    story = []
    page_width = doc.width

    metrics_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R2", "Mean Absolute Percent Error"],
            "Value": [
                external_metrics["RMSE"],
                external_metrics["MAE"],
                external_metrics["R2"],
                float(np.nanmean(external_results_df["Percent Error"])) if "Percent Error" in external_results_df.columns else np.nan,
            ],
        }
    )

    story.append(Paragraph("Consolidated Model Report", styles["Title"]))
    story.append(Spacer(1, 0.30 * cm))
    story.append(Paragraph(f"Model name: {model_name}", styles["Meta"]))
    story.append(Paragraph(f"Input variable: {input_col}", styles["Meta"]))
    story.append(Paragraph(f"Output variable: {output_col}", styles["Meta"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("External test metrics and comparison chart", styles["Heading2"]))
    story.append(Spacer(1, PDF_SETTINGS["small_gap_cm"] * cm))

    metrics_table = simple_table_from_df(
        metrics_df,
        max_rows=10,
        available_width=8.0 * cm,
        decimals=4,
        font_size=7.0,
        header_font_size=7.2,
    )
    ext_chart = Image(
        io.BytesIO(external_plot_bytes),
        width=page_width - 8.8 * cm,
        height=PDF_SETTINGS["consolidated_main_chart_height_cm"] * cm,
    )
    ext_chart.hAlign = "CENTER"

    top_layout = Table(
        [[metrics_table, ext_chart]],
        colWidths=[8.4 * cm, page_width - 8.4 * cm],
        hAlign="CENTER",
    )
    top_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    story.append(top_layout)
    story.append(PageBreak())

    story.append(Paragraph("External test percent error", styles["Heading2"]))
    story.append(Spacer(1, 0.18 * cm))
    err_chart = Image(
        io.BytesIO(external_error_plot_bytes),
        width=page_width,
        height=PDF_SETTINGS["consolidated_error_chart_height_cm"] * cm,
    )
    err_chart.hAlign = "CENTER"
    story.append(err_chart)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Cross-validation summary", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        simple_table_from_df(
            cv_summary,
            max_rows=10,
            available_width=page_width,
            decimals=4,
            font_size=6.8,
            header_font_size=7.0,
        )
    )
    story.append(Spacer(1, 0.30 * cm))

    story.append(Paragraph("External test detailed results", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        simple_table_from_df(
            external_results_df,
            max_rows=25,
            available_width=page_width,
            decimals=4,
            font_size=5.9,
            header_font_size=6.2,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def create_package_zip(
    model_name: str,
    bundle_filename: str,
    bundle_bytes: bytes,
    py_filename: str,
    py_code: str,
    txt_filename: str,
    txt_code: str,
    extras: Dict[str, bytes],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(bundle_filename, bundle_bytes)
        zf.writestr(py_filename, py_code)
        zf.writestr(txt_filename, txt_code)
        zf.writestr(
            "metadata.json",
            json.dumps(
                {
                    "model_name": model_name,
                    "created_at": datetime.now().isoformat(),
                    "contents": [bundle_filename, py_filename, txt_filename] + list(extras.keys()),
                },
                indent=2,
            ),
        )
        for name, data in extras.items():
            zf.writestr(name, data)
    buffer.seek(0)
    return buffer.getvalue()


def build_rejection_diagnostics(
    suspicious_df: pd.DataFrame,
    parameter_log: List[Dict],
    input_col: str,
) -> List[str]:
    messages = []

    if suspicious_df is None or suspicious_df.empty:
        return ["No suspicious runs were detected from the current external test table."]

    x_values = suspicious_df[input_col].to_numpy(dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_mean = float(np.mean(x_values))
    map_error = float(np.nanmean(suspicious_df["Percent Error"]))

    messages.append(f"Suspicious runs concentrate between {x_min:.6g} and {x_max:.6g} in the selected input range.")
    messages.append(f"The mean percent error of the suspicious subset is {map_error:.4f}%.")

    if x_mean <= np.nanpercentile(x_values, 35):
        messages.append("High-error runs are biased toward the lower input region; add Aspen points near the low-flow range.")
    elif x_mean >= np.nanpercentile(x_values, 65):
        messages.append("High-error runs are biased toward the upper input region; add Aspen points near the high-flow range.")
    else:
        messages.append("High-error runs are concentrated in the middle input region; improve sampling density around the central operating range.")

    param_df = pd.DataFrame(parameter_log) if parameter_log else pd.DataFrame()
    if not param_df.empty:
        worst_fold = param_df.sort_values("RMSE", ascending=False).iloc[0]
        messages.append(
            f"The weakest validation fold was Fold {int(worst_fold['Fold'])} with RMSE={float(worst_fold['RMSE']):.6f} and kernel {worst_fold['Kernel Name']}."
        )
        if not bool(worst_fold["Use WhiteKernel"]):
            messages.append("Consider enabling WhiteKernel or increasing alpha slightly to improve noise handling.")
    else:
        messages.append("Temporary parameter log is unavailable; review the external suspicious runs directly.")

    messages.append("Recommended countermeasures: inspect outliers, verify Aspen convergence, expand sampling in sparse regions, and compare Matern vs RBF if not already benchmarked.")
    return messages


@st.dialog("Welcome to Gaussian Model Generator", width="large")
def intro_dialog():
    st.markdown(
        """
        **Gaussian Model Generator** creates a 1D Gaussian Process surrogate model for PtMeOH studies from Aspen Plus Excel data.

        **How it works**
        1. It inspects the uploaded Aspen database.
        2. It detects candidate hydrogen-flow input columns.
        3. It detects the `Status` column when available.
        4. It lets you confirm one input and one output.
        5. It counts successful runs using `Status = OK`.
        6. It excludes non-converged or erroneous runs from the clean dataset.
        7. It builds a representative external test split.
        8. It performs 5-fold cross-validation on the training-validation data.
        9. It retrains the final production model on the full 80% development set.
        10. It evaluates the final model on the untouched external test set.
        11. It packages reports, tables, code, and the reusable model.

        **Required file**
        - Aspen Plus Excel file in `.xlsx`, `.xlsm`, or `.xls` format.

        **Outputs created**
        - Clean datasets.
        - External test and training-validation datasets.
        - Fold tables and plots.
        - Model metrics and parameter logs.
        - Training & Validation PDF report.
        - Optional consolidated final PDF report.
        - Reusable Python function, text summary, and serialized model package.
        """
    )
    if st.button("Initialize Gaussian Model Generator", type="primary", use_container_width=True):
        st.session_state["app_initialized"] = True
        add_log("analyzer", "Generator initialized successfully.")
        st.rerun()


def module_database_analyzer():
    render_module_header(
        "analyzer",
        "1. Database Analyzer",
        "Upload the Aspen Excel database, inspect the detected columns, and confirm the selected input and output variables.",
        "analyzer_notes",
    )

    uploaded = st.file_uploader(
        "Upload Aspen Plus Excel file",
        type=["xlsx", "xlsm", "xls"],
        key="uploader",
    )

    if uploaded is not None:
        st.session_state["uploaded_file_name"] = uploaded.name
        st.caption(f"Loaded file: {uploaded.name}")

    analyze_disabled = uploaded is None
    if st.button("Analyze Database", type="primary", use_container_width=True, disabled=analyze_disabled):
        try:
            set_module_status("analyzer", "running")
            add_log("analyzer", "Reading Aspen Excel file.")
            raw_df = read_uploaded_excel(uploaded)

            if raw_df.empty:
                raise ValueError("The uploaded Excel file is empty.")

            st.session_state["raw_df"] = raw_df
            st.session_state["status_column"] = find_status_column(raw_df)
            candidates = detect_input_candidates(list(raw_df.columns))
            st.session_state["detected_input_candidates"] = candidates
            set_module_status("analyzer", "ready")

            add_log("analyzer", f"File loaded successfully with {len(raw_df)} rows and {len(raw_df.columns)} columns.")
            if candidates:
                add_log("analyzer", f"Detected input candidates: {', '.join(map(str, candidates[:6]))}.")
            else:
                add_log("analyzer", "No strong input-column candidates were detected automatically.")

            if st.session_state["status_column"]:
                add_log("analyzer", f"Detected status column: {st.session_state['status_column']}.")
            else:
                add_log("analyzer", "No status column detected. The app will fall back to non-null logic only.")
        except Exception as exc:
            set_module_status("analyzer", "error")
            add_log("analyzer", f"Database analysis failed: {exc}")

    if st.session_state["raw_df"] is not None:
        raw_df = st.session_state["raw_df"]
        st.write("Detected columns")
        st.dataframe(pd.DataFrame({"Column": raw_df.columns.astype(str)}), use_container_width=True, height=220)

        candidate_options = st.session_state["detected_input_candidates"] or list(raw_df.columns)
        input_selection = st.selectbox(
            "Proposed input column",
            options=candidate_options,
            index=0,
            help="Candidate column should represent hydrogen flow.",
        )

        output_options = [col for col in raw_df.columns if col != input_selection]
        output_selection = st.selectbox(
            "Select output column",
            options=output_options,
            help="Target response to be modeled by the Gaussian surrogate.",
        )

        summary = summarize_runs(
            raw_df,
            input_selection,
            output_selection,
            st.session_state.get("status_column"),
        )
        st.session_state["run_summary"] = summary

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total runs", summary["total_runs"])
        c2.metric("Successful runs", summary["successful_runs"])
        c3.metric("Erroneous / non-converged", summary["erroneous_runs"])
        c4.metric("Successful [%]", f"{summary['ok_percentage']:.2f}")

        input_confirm = st.text_input("Confirm input column with Y/N", placeholder="Type Y or N", key="input_confirm")
        output_confirm = st.text_input("Confirm output column with Y/N", placeholder="Type Y or N", key="output_confirm")

        if st.session_state.get("status_column"):
            st.caption(f"Status-based filtering active using column: {st.session_state['status_column']}")

        if st.button("Confirm variable selections", use_container_width=True):
            input_ok = parse_yes_no(input_confirm)
            output_ok = parse_yes_no(output_confirm)

            if input_ok is True and output_ok is True:
                st.session_state["input_column"] = input_selection
                st.session_state["output_column"] = output_selection
                st.session_state["analyzer_done"] = True
                st.session_state["current_step"] = 2
                set_module_status("analyzer", "completed")
                set_module_status("cleaning", "ready")

                inspection_df = pd.DataFrame([summary])
                status_info = pd.DataFrame(
                    {
                        "Detected Status Column": [st.session_state.get("status_column") or "Not found"],
                        "Successful Status Value Used": ["OK"],
                    }
                )
                st.session_state["artifacts"]["database_inspection_summary.xlsx"] = to_excel_bytes(
                    {
                        "Summary": inspection_df,
                        "Detected Columns": pd.DataFrame({"Column": raw_df.columns.astype(str)}),
                        "Status Detection": status_info,
                    }
                )

                add_log("analyzer", f"Input column confirmed: {input_selection}.")
                add_log("analyzer", f"Output column confirmed: {output_selection}.")
                add_log("analyzer", "Database Analyzer completed successfully.")
                st.rerun()
            else:
                set_module_status("analyzer", "ready")
                add_log("analyzer", "Variable confirmation failed. Confirm both selections with Y before proceeding.")

        if "database_inspection_summary.xlsx" in st.session_state["artifacts"]:
            st.download_button(
                "Download inspection summary",
                data=st.session_state["artifacts"]["database_inspection_summary.xlsx"],
                file_name="database_inspection_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        if st.button(
            "Proceed to Data Cleaning & Preparation",
            disabled=not st.session_state["analyzer_done"],
            use_container_width=True,
            key="proceed_m1",
        ):
            st.session_state["current_step"] = 2
            st.rerun()

    render_log("analyzer", "analyzer_log_view")


def module_cleaning_preparation():
    render_module_header(
        "cleaning",
        "2. Data Cleaning & Preparation",
        "Remove erroneous rows, create the clean database, and generate the representative 80/20 development-test split.",
        "cleaning_notes",
    )

    if st.session_state["current_step"] < 2 or not st.session_state["analyzer_done"]:
        show_locked_message("This module is locked. Complete Database Analyzer first.")
        render_log("cleaning", "cleaning_log_locked")
        return

    if st.button("Run Data Cleaning & Preparation", type="primary", use_container_width=True):
        try:
            set_module_status("cleaning", "running")
            raw_df = st.session_state["raw_df"].copy()
            input_col = st.session_state["input_column"]
            output_col = st.session_state["output_column"]
            status_col = st.session_state.get("status_column")

            clean_df = raw_df.copy()
            total_before = len(clean_df)

            if status_col and status_col in clean_df.columns:
                add_log("cleaning", f"Filtering rows using {status_col} == OK.")
                clean_df = clean_df[normalize_status(clean_df[status_col]) == "ok"].copy()
            else:
                add_log("cleaning", "Status column not available. Falling back to non-null input/output filtering only.")

            add_log("cleaning", "Converting selected input and output columns to numeric form.")
            clean_df[input_col] = pd.to_numeric(clean_df[input_col], errors="coerce")
            clean_df[output_col] = pd.to_numeric(clean_df[output_col], errors="coerce")

            clean_df = clean_df[clean_df[input_col].notna()].copy()
            clean_df = clean_df[clean_df[output_col].notna()].copy()
            clean_df = clean_df.drop_duplicates(subset=[input_col, output_col]).sort_values(input_col).reset_index(drop=True)

            removed = total_before - len(clean_df)

            if len(clean_df) < 6:
                raise ValueError("The clean dataset is too small. Provide more valid Aspen runs.")

            train_val_df, external_test_df = representative_split(clean_df, input_col=input_col, test_frac=0.2)

            st.session_state["clean_df"] = clean_df
            st.session_state["train_val_df"] = train_val_df
            st.session_state["external_test_df"] = external_test_df
            st.session_state["cleaning_done"] = True
            st.session_state["current_step"] = 3

            st.session_state["artifacts"]["clean_data.xlsx"] = to_excel_bytes({"Clean Data": clean_df})
            st.session_state["artifacts"]["external_data_test.xlsx"] = to_excel_bytes({"External Data Test": external_test_df})
            st.session_state["artifacts"]["data_training_validation_set.xlsx"] = to_excel_bytes({"TrainingValidation": train_val_df})

            set_module_status("cleaning", "completed")
            set_module_status("training", "ready")

            add_log("cleaning", f"Removed {removed} invalid, non-converged, or duplicated rows.")
            add_log("cleaning", f"Representative split created: {len(train_val_df)} rows for training-validation and {len(external_test_df)} rows for external test.")
            st.rerun()
        except Exception as exc:
            set_module_status("cleaning", "error")
            add_log("cleaning", f"Data Cleaning & Preparation failed: {exc}")

    if st.session_state["cleaning_done"]:
        clean_df = st.session_state["clean_df"]
        train_val_df = st.session_state["train_val_df"]
        external_test_df = st.session_state["external_test_df"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Clean rows", len(clean_df))
        c2.metric("Training-validation rows", len(train_val_df))
        c3.metric("External test rows", len(external_test_df))

        st.write("Clean dataset preview")
        st.dataframe(clean_df.head(20), use_container_width=True)

        st.download_button(
            "Download Clean Data",
            data=st.session_state["artifacts"]["clean_data.xlsx"],
            file_name="clean_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Download External Data Test",
            data=st.session_state["artifacts"]["external_data_test.xlsx"],
            file_name="external_data_test.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Download Data Training and Validation Set",
            data=st.session_state["artifacts"]["data_training_validation_set.xlsx"],
            file_name="data_training_validation_set.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if st.button(
            "Proceed to Training & Validation",
            disabled=not st.session_state["cleaning_done"],
            use_container_width=True,
            key="proceed_m2",
        ):
            st.session_state["current_step"] = 3
            st.rerun()

    render_log("cleaning", "cleaning_log_view")


def module_training_validation():
    render_module_header(
        "training",
        "3. Training & Validation",
        "Execute 5-fold cross-validation, benchmark optional kernels, export fold-level artifacts, and retrain the final production model.",
        "training_notes",
    )

    if st.session_state["current_step"] < 3 or not st.session_state["cleaning_done"]:
        show_locked_message("This module is locked. Complete Data Cleaning & Preparation first.")
        render_log("training", "training_log_locked")
        return

    kernel_mode = st.selectbox(
        "Kernel mode",
        options=["Matern 2.5 (default)", "RBF benchmark", "Compare Matern vs RBF"],
    )
    use_white_kernel = st.checkbox("Use WhiteKernel for explicit noise handling", value=False)
    alpha_value = st.number_input("Alpha regularization", min_value=1e-10, max_value=1e-2, value=1e-8, format="%.1e")
    model_name = st.text_input("Model name", value=st.session_state.get("model_name", "PtMeOH_GP_Model"))
    st.session_state["model_name"] = model_name.strip() or "PtMeOH_GP_Model"

    if st.button("Run Training & Validation", type="primary", use_container_width=True):
        try:
            set_module_status("training", "running")
            add_log("training", "Starting 5-fold cross-validation.")
            df = st.session_state["train_val_df"].copy()
            input_col = st.session_state["input_column"]
            output_col = st.session_state["output_column"]

            if len(df) < 5:
                raise ValueError("The training-validation dataset must contain at least 5 rows for 5-fold CV.")

            comparison_df = None
            if kernel_mode == "Matern 2.5 (default)":
                chosen_cv = perform_cv(df, input_col, output_col, "Matern 2.5", use_white_kernel, alpha_value)
            elif kernel_mode == "RBF benchmark":
                chosen_cv = perform_cv(df, input_col, output_col, "RBF", use_white_kernel, alpha_value)
            else:
                add_log("training", "Running kernel comparison between Matern 2.5 and RBF.")
                result_matern = perform_cv(df, input_col, output_col, "Matern 2.5", use_white_kernel, alpha_value, enable_logs=False)
                result_rbf = perform_cv(df, input_col, output_col, "RBF", use_white_kernel, alpha_value, enable_logs=False)

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Kernel": result_matern["kernel_name"],
                            "Mean RMSE": float(result_matern["summary_df"].loc[result_matern["summary_df"]["Metric"] == "RMSE", "Mean"].iloc[0]),
                            "Mean MAE": float(result_matern["summary_df"].loc[result_matern["summary_df"]["Metric"] == "MAE", "Mean"].iloc[0]),
                            "Mean R2": float(result_matern["summary_df"].loc[result_matern["summary_df"]["Metric"] == "R2", "Mean"].iloc[0]),
                        },
                        {
                            "Kernel": result_rbf["kernel_name"],
                            "Mean RMSE": float(result_rbf["summary_df"].loc[result_rbf["summary_df"]["Metric"] == "RMSE", "Mean"].iloc[0]),
                            "Mean MAE": float(result_rbf["summary_df"].loc[result_rbf["summary_df"]["Metric"] == "MAE", "Mean"].iloc[0]),
                            "Mean R2": float(result_rbf["summary_df"].loc[result_rbf["summary_df"]["Metric"] == "R2", "Mean"].iloc[0]),
                        },
                    ]
                )
                chosen_cv = select_best_cv_result(result_matern, result_rbf)
                add_log("training", f"Kernel comparison completed. Selected kernel: {chosen_cv['kernel_name']}.")

            final_model = GaussianSurrogate1D(
                kernel_name=chosen_cv["kernel_name"],
                use_white_kernel=use_white_kernel,
                alpha=alpha_value,
            ).fit(df[[input_col]].to_numpy(), df[output_col].to_numpy())

            model_bundle = final_model.export_bundle(
                model_name=st.session_state["model_name"],
                input_column=input_col,
                output_column=output_col,
            )

            st.session_state["trained_model"] = final_model
            st.session_state["trained_model_bundle"] = model_bundle
            st.session_state["trained_model_metadata"] = {
                "Model Name": st.session_state["model_name"],
                "Kernel Name": chosen_cv["kernel_name"],
                "Use WhiteKernel": use_white_kernel,
                "Alpha": alpha_value,
                "Input Column": input_col,
                "Output Column": output_col,
                "Fitted Kernel": final_model.fitted_kernel,
                "Created At": datetime.now().isoformat(),
            }

            st.session_state["cv_fold_results"] = chosen_cv["metrics_df"].to_dict(orient="records")
            st.session_state["cv_summary"] = chosen_cv["summary_df"]
            st.session_state["cv_comparison"] = comparison_df
            st.session_state["temporary_parameter_log"] = chosen_cv["parameter_log"]

            safe_name = st.session_state["model_name"].replace(" ", "_")
            bundle_filename = f"{safe_name}.joblib"
            py_filename = f"{safe_name}.py"
            txt_filename = f"{safe_name}.txt"

            bundle_bytes = joblib_bytes(model_bundle)
            python_code = build_python_wrapper(bundle_filename)
            text_code = build_text_summary(
                st.session_state["model_name"],
                input_col,
                output_col,
                chosen_cv["summary_df"],
            )

            cv_book = {
                **chosen_cv["fold_tables"],
                "CV Metrics": chosen_cv["metrics_df"],
                "CV Summary": chosen_cv["summary_df"],
                "Parameter Log": pd.DataFrame(chosen_cv["parameter_log"]),
            }
            if comparison_df is not None:
                cv_book["Kernel Comparison"] = comparison_df

            st.session_state["artifacts"]["cv_results.xlsx"] = to_excel_bytes(cv_book)
            st.session_state["artifacts"]["cv_metrics_plot.png"] = chosen_cv["cv_metrics_plot"]
            st.session_state["artifacts"]["training_validation_report.pdf"] = build_training_pdf(
                st.session_state["model_name"],
                input_col,
                output_col,
                chosen_cv,
                comparison_df,
            )
            st.session_state["artifacts"]["model_parameters.xlsx"] = to_excel_bytes(
                {
                    "Model Metadata": pd.DataFrame([st.session_state["trained_model_metadata"]]),
                    "Temporary Parameter Log": pd.DataFrame(chosen_cv["parameter_log"]),
                }
            )
            st.session_state["artifacts"][bundle_filename] = bundle_bytes
            st.session_state["artifacts"][py_filename] = python_code.encode("utf-8")
            st.session_state["artifacts"][txt_filename] = text_code.encode("utf-8")

            for name, data in chosen_cv["fold_prediction_plots"].items():
                st.session_state["artifacts"][name] = data
            for name, data in chosen_cv["fold_error_plots"].items():
                st.session_state["artifacts"][name] = data

            st.session_state["training_done"] = True
            st.session_state["current_step"] = 4
            set_module_status("training", "completed")
            set_module_status("testing", "ready")
            add_log("training", "Final production model retrained on the full training-validation dataset.")
            add_log("training", "Training & Validation module completed successfully.")
            st.rerun()
        except Exception as exc:
            set_module_status("training", "error")
            add_log("training", f"Training & Validation failed: {exc}")

    if st.session_state["training_done"]:
        st.write("Cross-validation summary")
        st.dataframe(st.session_state["cv_summary"], use_container_width=True)

        if st.session_state["cv_comparison"] is not None:
            st.write("Kernel comparison")
            st.dataframe(st.session_state["cv_comparison"], use_container_width=True)

        st.image(st.session_state["artifacts"]["cv_metrics_plot.png"], caption="5-fold CV metrics")

        st.download_button(
            "Download Training & Validation Report (PDF)",
            data=st.session_state["artifacts"]["training_validation_report.pdf"],
            file_name="training_validation_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "Download Consolidated CV Results (Excel)",
            data=st.session_state["artifacts"]["cv_results.xlsx"],
            file_name="cv_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Download Model Parameters (Excel)",
            data=st.session_state["artifacts"]["model_parameters.xlsx"],
            file_name="model_parameters.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if st.button(
            "Proceed to Test & Packing",
            disabled=not st.session_state["training_done"],
            use_container_width=True,
            key="proceed_m3",
        ):
            st.session_state["current_step"] = 4
            st.rerun()

    render_log("training", "training_log_view")


def module_test_packing():
    render_module_header(
        "testing",
        "4. Test & Packing",
        "Evaluate the final model on the untouched external test set, generate the final diagnostics, and package the reusable model if approved.",
        "testing_notes",
    )

    if st.session_state["current_step"] < 4 or not st.session_state["training_done"]:
        show_locked_message("This module is locked. Complete Training & Validation first.")
        render_log("testing", "testing_log_locked")
        return

    generate_consolidated_pdf = st.checkbox("Generate final consolidated PDF report", value=True)
    satisfaction = st.text_input(
        "Are you satisfied with the model performance? (Yes/No)",
        placeholder="Type Yes or No",
    )

    if st.button("Run External Test & Packing", type="primary", use_container_width=True):
        try:
            set_module_status("testing", "running")
            add_log("testing", "Starting external test evaluation.")
            model = st.session_state["trained_model"]
            input_col = st.session_state["input_column"]
            output_col = st.session_state["output_column"]
            test_df = st.session_state["external_test_df"].copy()

            X_test = test_df[[input_col]].astype(float).to_numpy()
            y_test = test_df[output_col].astype(float).to_numpy()

            y_pred, y_std = model.predict(X_test, return_std=True)
            metrics = compute_metrics(y_test, y_pred)
            percent_error = safe_percent_error(y_test, y_pred)

            results_df = pd.DataFrame(
                {
                    input_col: X_test.flatten(),
                    output_col: y_test,
                    "Prediction": y_pred,
                    "Predictive Std": y_std,
                    "Absolute Error": np.abs(y_test - y_pred),
                    "Percent Error": percent_error,
                }
            ).sort_values(input_col).reset_index(drop=True)

            comparison_df = results_df[[input_col, output_col, "Prediction", "Predictive Std"]].copy()
            error_df = results_df[[input_col, "Absolute Error", "Percent Error"]].copy()
            suspicious_df = results_df.sort_values("Absolute Error", ascending=False).head(min(10, len(results_df))).reset_index(drop=True)

            external_plot = create_external_comparison_plot(results_df, input_col, output_col)
            external_error_plot = create_external_error_plot(results_df, input_col)

            st.session_state["external_test_results"] = {
                "metrics": metrics,
                "results_df": results_df,
                "comparison_df": comparison_df,
                "error_df": error_df,
            }
            st.session_state["suspicious_points"] = suspicious_df
            st.session_state["diagnostics_text"] = build_rejection_diagnostics(
                suspicious_df,
                st.session_state["temporary_parameter_log"],
                input_col,
            )

            st.session_state["artifacts"]["external_test_results.xlsx"] = to_excel_bytes(
                {
                    "Comparison Table": comparison_df,
                    "Error Table": error_df,
                    "External Metrics": pd.DataFrame([metrics]),
                    "Suspicious Runs": suspicious_df,
                }
            )
            st.session_state["artifacts"]["external_test_plot.png"] = external_plot
            st.session_state["artifacts"]["external_test_error_plot.png"] = external_error_plot

            if generate_consolidated_pdf:
                st.session_state["artifacts"]["consolidated_model_report.pdf"] = build_consolidated_pdf(
                    st.session_state["model_name"],
                    input_col,
                    output_col,
                    st.session_state["cv_summary"],
                    results_df,
                    metrics,
                    external_plot,
                    external_error_plot,
                )

            decision = parse_yes_no(satisfaction)
            safe_name = st.session_state["model_name"].replace(" ", "_")
            bundle_filename = f"{safe_name}.joblib"
            py_filename = f"{safe_name}.py"
            txt_filename = f"{safe_name}.txt"

            if decision is True:
                bundle_bytes = st.session_state["artifacts"][bundle_filename]
                py_code = st.session_state["artifacts"][py_filename].decode("utf-8")
                txt_code = build_text_summary(
                    st.session_state["model_name"],
                    input_col,
                    output_col,
                    st.session_state["cv_summary"],
                    metrics,
                )
                st.session_state["artifacts"][txt_filename] = txt_code.encode("utf-8")

                extras = {
                    "clean_data.xlsx": st.session_state["artifacts"]["clean_data.xlsx"],
                    "external_data_test.xlsx": st.session_state["artifacts"]["external_data_test.xlsx"],
                    "data_training_validation_set.xlsx": st.session_state["artifacts"]["data_training_validation_set.xlsx"],
                    "model_parameters.xlsx": st.session_state["artifacts"]["model_parameters.xlsx"],
                    "cv_results.xlsx": st.session_state["artifacts"]["cv_results.xlsx"],
                    "training_validation_report.pdf": st.session_state["artifacts"]["training_validation_report.pdf"],
                    "external_test_results.xlsx": st.session_state["artifacts"]["external_test_results.xlsx"],
                    "external_test_plot.png": st.session_state["artifacts"]["external_test_plot.png"],
                    "external_test_error_plot.png": st.session_state["artifacts"]["external_test_error_plot.png"],
                }
                if "consolidated_model_report.pdf" in st.session_state["artifacts"]:
                    extras["consolidated_model_report.pdf"] = st.session_state["artifacts"]["consolidated_model_report.pdf"]

                st.session_state["artifacts"]["final_model_package.zip"] = create_package_zip(
                    st.session_state["model_name"],
                    bundle_filename,
                    bundle_bytes,
                    py_filename,
                    py_code,
                    txt_filename,
                    txt_code,
                    extras,
                )
                st.session_state["final_approved"] = True
                add_log("testing", "User approved model performance. Final model package created.")
            elif decision is False:
                st.session_state["final_approved"] = False
                add_log("testing", "User rejected current performance. Diagnostic recommendations were generated.")
            else:
                st.session_state["final_approved"] = False
                add_log("testing", "External testing completed, but the approval answer was not recognized. Please respond with Yes or No.")

            st.session_state["testing_done"] = True
            set_module_status("testing", "completed")
            add_log("testing", f"External test completed. RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}.")
            st.rerun()
        except Exception as exc:
            set_module_status("testing", "error")
            add_log("testing", f"Test & Packing failed: {exc}")

    if st.session_state["testing_done"] and st.session_state["external_test_results"] is not None:
        results_pack = st.session_state["external_test_results"]
        metrics = results_pack["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{metrics['RMSE']:.6f}")
        c2.metric("MAE", f"{metrics['MAE']:.6f}")
        c3.metric("R²", f"{metrics['R2']:.6f}")

        st.image(st.session_state["artifacts"]["external_test_plot.png"], caption="External test comparison")
        st.image(st.session_state["artifacts"]["external_test_error_plot.png"], caption="External test percent error")

        st.write("Comparison table")
        st.dataframe(results_pack["comparison_df"], use_container_width=True)

        st.write("Error table")
        st.dataframe(results_pack["error_df"], use_container_width=True)

        st.download_button(
            "Download External Test Results (Excel)",
            data=st.session_state["artifacts"]["external_test_results.xlsx"],
            file_name="external_test_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if "consolidated_model_report.pdf" in st.session_state["artifacts"]:
            st.download_button(
                "Download Consolidated Model Report (PDF)",
                data=st.session_state["artifacts"]["consolidated_model_report.pdf"],
                file_name="consolidated_model_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        if st.session_state["final_approved"]:
            st.success("Model approved. Final package is available below and in the sidebar.")
            st.download_button(
                "Download Final Model Package (.zip)",
                data=st.session_state["artifacts"]["final_model_package.zip"],
                file_name="final_model_package.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.warning("Model not approved yet.")
            st.write("Suspicious runs with the highest absolute error")
            st.dataframe(st.session_state["suspicious_points"], use_container_width=True)

            st.write("Suggested countermeasures")
            for item in st.session_state["diagnostics_text"]:
                st.markdown(f"- {item}")

            restart_choice = st.text_input(
                "Do you want to restart the full generator with a new Aspen database? (Yes/No)",
                key="restart_choice",
            )
            if parse_yes_no(restart_choice) is True:
                reset_workflow()

    render_log("testing", "testing_log_view")


def render_final_sidebar():
    with st.sidebar:
        st.header("Final Download Center")

        if st.button("Reset workflow", use_container_width=True):
            reset_workflow()

        if not st.session_state["testing_done"]:
            st.info("Downloads will be listed here after external testing is completed.")
            return

        preferred_order = [
            "clean_data.xlsx",
            "external_data_test.xlsx",
            "data_training_validation_set.xlsx",
            "cv_results.xlsx",
            "model_parameters.xlsx",
            "training_validation_report.pdf",
            "external_test_results.xlsx",
            "consolidated_model_report.pdf",
            "final_model_package.zip",
        ]

        for name in preferred_order:
            if name in st.session_state["artifacts"]:
                mime = "application/octet-stream"
                if name.endswith(".xlsx"):
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif name.endswith(".pdf"):
                    mime = "application/pdf"
                elif name.endswith(".zip"):
                    mime = "application/zip"
                elif name.endswith(".png"):
                    mime = "image/png"
                elif name.endswith(".joblib"):
                    mime = "application/octet-stream"
                elif name.endswith(".py") or name.endswith(".txt"):
                    mime = "text/plain"

                st.download_button(
                    f"Download {name}",
                    data=st.session_state["artifacts"][name],
                    file_name=name,
                    mime=mime,
                    use_container_width=True,
                    key=f"sidebar_{name}",
                )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    inject_styles()

    if not st.session_state["app_initialized"]:
        intro_dialog()

    render_header()
    render_final_sidebar()

    tabs = st.tabs(
        [
            "1. Database Analyzer",
            "2. Data Cleaning & Preparation",
            "3. Training & Validation",
            "4. Test & Packing",
        ]
    )

    with tabs[0]:
        module_database_analyzer()
    with tabs[1]:
        module_cleaning_preparation()
    with tabs[2]:
        module_training_validation()
    with tabs[3]:
        module_test_packing()

if __name__ == "__main__":
    main()
