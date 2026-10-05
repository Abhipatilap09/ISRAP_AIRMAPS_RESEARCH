"""Experiment History — view, compare, reproduce, and export past experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.experiment_store import (
    delete_experiment,
    list_experiments,
    update_notes,
)
from ui.theme import COLORS, inject_css, info_box, metric_card


def _parse_metrics(m):
    if isinstance(m, dict):
        return m
    if isinstance(m, str):
        try:
            return json.loads(m)
        except Exception:
            return {}
    return {}


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    st.set_page_config(
        page_title="Experiment History | ISRAP AIRMAPS",
        layout="wide",
        page_icon="🕐",
    )
    inject_css()

    st.markdown(
        f"<h1 style='color:{COLORS['navy']};'>🕐 Experiment History</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Every anomaly detection and imputation experiment is automatically recorded. "
        "Review, compare, add notes, or export any past run."
    )

    raw_exps = list_experiments(limit=500)

    if not raw_exps:
        st.markdown(
            f"""<div class="empty-state">
            <div class="empty-state-icon">🕐</div>
            <div class="empty-state-title">No experiments recorded yet</div>
            <div style='color:{COLORS["muted"]};'>
                Run anomaly detection on the <b>Anomaly Lab</b> page or
                imputation on the <b>Imputation Lab</b> page to see results here.
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(raw_exps)
    df["metrics_dict"] = df["metrics"].apply(_parse_metrics)
    df["n_flagged"] = df["metrics_dict"].apply(lambda m: m.get("n_flagged", None))
    df["pct_flagged"] = df["metrics_dict"].apply(lambda m: m.get("pct_flagged", None))
    df["mae"] = df["metrics_dict"].apply(lambda m: m.get("mae", None))
    df["r2"] = df["metrics_dict"].apply(lambda m: m.get("r2", None))
    df["runtime_disp"] = df["runtime_sec"].apply(
        lambda v: f"{_safe_float(v, 0):.2f}s" if v is not None else "—"
    )
    df["created_disp"] = pd.to_datetime(df["created_at"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    total = len(df)
    stations_used = df["station"].nunique()
    methods_used = df["method"].nunique()
    ok_count = int((df["status"] == "ok").sum()) if "status" in df.columns else total

    cols = st.columns(5)
    cols[0].markdown(metric_card("Total Experiments", str(total)), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Stations Used", str(stations_used)), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Pollutants", str(df["pollutant"].nunique())), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Methods", str(methods_used)), unsafe_allow_html=True)
    cols[4].markdown(metric_card("Successful", str(ok_count), color=COLORS["green"]),
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_list, tab_charts, tab_compare, tab_manage = st.tabs([
        "Experiment List", "Analytics", "Compare", "Manage"
    ])

    with tab_list:
        fc1, fc2, fc3 = st.columns(3)
        station_opts = ["All"] + sorted(df["station"].dropna().unique().tolist())
        pol_opts = ["All"] + sorted(df["pollutant"].dropna().unique().tolist())
        method_opts = ["All"] + sorted(df["method"].dropna().unique().tolist())
        with fc1:
            f_station = st.selectbox("Station", station_opts)
        with fc2:
            f_pol = st.selectbox("Pollutant", pol_opts)
        with fc3:
            f_meth = st.selectbox("Method", method_opts)

        fdf = df.copy()
        if f_station != "All":
            fdf = fdf[fdf["station"] == f_station]
        if f_pol != "All":
            fdf = fdf[fdf["pollutant"] == f_pol]
        if f_meth != "All":
            fdf = fdf[fdf["method"] == f_meth]

        st.caption(f"Showing {len(fdf)} of {total} experiments")
        show_cols = ["id", "created_disp", "station", "pollutant", "method",
                     "n_flagged", "pct_flagged", "mae", "r2", "runtime_disp"]
        if "status" in fdf.columns:
            show_cols.append("status")
        if "notes" in fdf.columns:
            show_cols.append("notes")
        show_cols = [c for c in show_cols if c in fdf.columns]
        st.dataframe(fdf[show_cols].head(200), use_container_width=True, hide_index=True)

        csv_bytes = fdf[show_cols].to_csv(index=False).encode()
        st.download_button("📥 Export filtered list (CSV)", csv_bytes,
                           "experiment_history_filtered.csv", mime="text/csv")

    with tab_charts:
        col_a, col_b = st.columns(2)
        with col_a:
            method_counts = df.groupby("method").size().reset_index(name="count")
            method_counts = method_counts.sort_values("count", ascending=True)
            fig_m = go.Figure(go.Bar(
                x=method_counts["count"], y=method_counts["method"],
                orientation="h", marker_color=COLORS["teal"],
            ))
            fig_m.update_layout(title="Experiments by Method", template="plotly_white",
                                height=300, margin=dict(l=140, r=20, t=40, b=40))
            st.plotly_chart(fig_m, use_container_width=True)
        with col_b:
            sc = df.groupby("station").size().reset_index(name="count")
            fig_s = go.Figure(go.Pie(labels=sc["station"], values=sc["count"], hole=0.4))
            fig_s.update_layout(title="Experiments by Station", template="plotly_white",
                                height=300, margin=dict(l=20, r=20, t=40, b=40))
            st.plotly_chart(fig_s, use_container_width=True)

        if "created_at" in df.columns:
            df["created_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
            df_ts = df.dropna(subset=["created_dt"]).copy()
            if not df_ts.empty:
                df_ts["date"] = df_ts["created_dt"].dt.date
                daily = df_ts.groupby("date").size().reset_index(name="count")
                fig_ts = go.Figure(go.Bar(
                    x=daily["date"].astype(str), y=daily["count"],
                    marker_color=COLORS["blue"],
                ))
                fig_ts.update_layout(
                    title="Experiment Volume Over Time", template="plotly_white",
                    height=250, margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

        pct_data = df.copy()
        pct_data["pct_flagged_num"] = pd.to_numeric(pct_data["pct_flagged"], errors="coerce")
        pct_agg = pct_data.dropna(subset=["pct_flagged_num"]).groupby("method")["pct_flagged_num"].mean().reset_index()
        if not pct_agg.empty:
            pct_agg = pct_agg.sort_values("pct_flagged_num", ascending=True)
            fig_pct = go.Figure(go.Bar(
                x=pct_agg["pct_flagged_num"], y=pct_agg["method"],
                orientation="h", marker_color=COLORS["red"],
            ))
            fig_pct.update_layout(
                title="Average % Flagged by Method", template="plotly_white",
                height=max(200, len(pct_agg) * 30 + 80),
                margin=dict(l=160, r=20, t=40, b=40),
                xaxis_title="Mean % Flagged",
            )
            st.plotly_chart(fig_pct, use_container_width=True)

    with tab_compare:
        info_box(
            "Select two experiment IDs to compare their parameters and metrics side-by-side.",
            kind="info",
        )
        all_ids = df["id"].tolist()
        if len(all_ids) < 2:
            st.info("Need at least 2 experiments to compare.")
        else:
            c_a, c_b = st.columns(2)
            with c_a:
                id_a = st.selectbox("Experiment A", all_ids, key="cmp_a")
            with c_b:
                id_b = st.selectbox(
                    "Experiment B", [i for i in all_ids if i != id_a], key="cmp_b"
                )
            row_a = df[df["id"] == id_a].iloc[0] if not df[df["id"] == id_a].empty else None
            row_b = df[df["id"] == id_b].iloc[0] if not df[df["id"] == id_b].empty else None
            if row_a is not None and row_b is not None:
                compare_fields = ["station", "pollutant", "method", "created_disp",
                                  "runtime_disp", "status", "n_flagged", "pct_flagged",
                                  "mae", "r2"]
                data = []
                for field in compare_fields:
                    va = str(row_a.get(field, "—") or "—")
                    vb = str(row_b.get(field, "—") or "—")
                    diff = "✅" if va == vb else "≠"
                    data.append({"Field": field, "Experiment A": va, "Match": diff, "Experiment B": vb})
                params_a = _parse_metrics(row_a.get("params", {}))
                params_b = _parse_metrics(row_b.get("params", {}))
                for k in sorted(set(list(params_a) + list(params_b))):
                    va = str(params_a.get(k, "—"))
                    vb = str(params_b.get(k, "—"))
                    data.append({"Field": f"param.{k}", "Experiment A": va,
                                 "Match": "✅" if va == vb else "≠", "Experiment B": vb})
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    with tab_manage:
        col_note, col_del = st.columns(2)
        with col_note:
            st.markdown("**Add / Update Note**")
            note_id = st.text_input("Experiment ID", key="note_id")
            note_text = st.text_area("Note", key="note_text", height=80)
            if st.button("Save Note"):
                if note_id:
                    update_notes(note_id, note_text)
                    st.success(f"Note saved for {note_id}")
                    st.rerun()
                else:
                    st.warning("Enter an experiment ID.")
        with col_del:
            st.markdown("**Delete Experiment**")
            del_id = st.text_input("Experiment ID to delete", key="del_id")
            confirm = st.checkbox("Confirm deletion", key="del_confirm")
            if st.button("Delete", type="secondary"):
                if del_id and confirm:
                    delete_experiment(del_id)
                    st.success(f"Deleted {del_id}")
                    st.rerun()
                elif not del_id:
                    st.warning("Enter an experiment ID.")
                else:
                    st.warning("Check the confirmation box.")

        st.divider()
        full_csv = df.drop(columns=["metrics_dict", "created_dt"], errors="ignore").to_csv(index=False).encode()
        st.download_button("📥 Export all (CSV)", full_csv,
                           "all_experiments.csv", mime="text/csv")
        full_json = json.dumps(raw_exps, default=str, indent=2).encode()
        st.download_button("📥 Export all (JSON)", full_json,
                           "all_experiments.json", mime="application/json")


if __name__ == "__main__":
    main()

