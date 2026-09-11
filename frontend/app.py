"""
Streamlit Dashboard – LLM Evaluation Dashboard
================================================
Provides:
  • 🧪 Mode 1: Model Comparison (Multi-Model evaluation, AI Judge, Ground Truth)
  • ⚖️ Mode 2: Prompt A/B Testing (Side-by-side prompt engineering & delta telemetry)
  • 📈 Historical Analytics & CSV Export
"""

import os
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
AVAILABLE_MODELS = ["GPT-4o", "Gemini 3.6", "Claude 3.5 Sonnet", "Claude 3 Opus", "Local Llama"]
NEON_PALETTE = ["#8A2BE2", "#FF00FF", "#39FF14", "#FFEA00", "#00F0FF", "#FF007F"]

st.set_page_config(
    page_title="NEURAL EVAL CORE // LLM Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper – call backend
# ---------------------------------------------------------------------------

def _post_evaluate(
    prompt: str,
    models: list[str],
    prompt_version: str = "v1",
    ground_truth: str | None = None,
) -> list[dict]:
    """Send an evaluation request to the FastAPI backend."""
    with httpx.Client(timeout=45.0) as client:
        resp = client.post(
            f"{API_BASE}/evaluate",
            json={
                "prompt": prompt,
                "models": models,
                "prompt_version": prompt_version,
                "ground_truth": ground_truth,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _get_results(limit: int = 200) -> list[dict]:
    """Fetch historical results from the backend."""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{API_BASE}/results", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()


def _status_badge(row: pd.Series, min_latency: float) -> str:
    """Return a human-friendly status label based on the model's strengths."""
    avg_score = (row["correctness_score"] + row["relevance_score"] + row["faithfulness_score"]) / 3
    if row["estimated_cost_usd"] == 0:
        return "💰 Termurah"
    if row["latency_seconds"] == min_latency:
        return "⚡ Tercepat"
    if avg_score >= 0.90:
        return "✅ Optimal"
    return "🔄 Normal"


# ---------------------------------------------------------------------------
# Custom CSS Styling (Cyberpunk / AI Futuristic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Space+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}
.glow-title {
    background: linear-gradient(90deg, #8A2BE2, #FF00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 0px;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.subtitle {
    color: #8b9bb4;
    font-family: 'Space Mono', monospace;
    font-size: 0.9em;
    margin-bottom: 25px;
}
.stButton>button {
    background: linear-gradient(45deg, #8A2BE2, #FF00FF);
    color: white;
    border: none;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.4);
    border-radius: 6px;
    font-weight: bold;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    box-shadow: 0 0 25px rgba(255, 0, 255, 0.7);
    transform: scale(1.01);
}
.stTextArea textarea {
    background-color: #0d1424 !important;
    border: 1px solid #8A2BE2 !important;
    color: #E0E7FF !important;
    font-family: 'Space Mono', monospace !important;
    box-shadow: inset 0 0 10px rgba(138, 43, 226, 0.15);
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #D18BFF;
}
.judge-box {
    background-color: rgba(138, 43, 226, 0.12);
    border-left: 4px solid #8A2BE2;
    padding: 12px 16px;
    border-radius: 4px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 0.92em;
    color: #E0E7FF;
}
.winner-card {
    background: linear-gradient(90deg, rgba(138, 43, 226, 0.25), rgba(255, 0, 255, 0.15));
    border: 1px solid #FF00FF;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prompt Version Definitions & Metadata
# ---------------------------------------------------------------------------
VERSION_MAP = {
    "v1": {
        "title": "v1 - Direct (Raw Prompt)",
        "badge": "⚡ v1: Direct Raw",
        "desc": "Mengirimkan prompt asli secara langsung tanpa modifikasi instruksi sistem tambahan.",
    },
    "v2": {
        "title": "v2 - Expert Persona & Structured",
        "badge": "🎓 v2: Expert Persona",
        "desc": "Instruksi Senior Domain Expert dengan output terstruktur, poin-poin utama, dan contoh konkret.",
    },
    "v3": {
        "title": "v3 - Chain-of-Thought & Concise",
        "badge": "🧠 v3: Step-by-Step CoT",
        "desc": "Instruksi penalaran bertahap (step-by-step) analitis, kritis, dan to-the-point tanpa basa-basi.",
    },
}

# ---------------------------------------------------------------------------
# Sidebar – Mode Selection & Global Settings
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ CONTROL PANEL")

app_mode = st.sidebar.radio(
    "Pilih Mode Operasi:",
    ["🧪 Model Comparison", "⚖️ Prompt A/B Testing"],
    index=0,
)

st.sidebar.divider()

if app_mode == "🧪 Model Comparison":
    selected_models = st.sidebar.multiselect(
        "Pilih Model Evaluasi",
        AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
    )
    prompt_version = st.sidebar.selectbox(
        "Versi Prompt Engineering",
        options=list(VERSION_MAP.keys()),
        format_func=lambda k: VERSION_MAP[k]["title"],
        index=0,
    )
    st.sidebar.info(f"**{VERSION_MAP[prompt_version]['badge']}**\n\n{VERSION_MAP[prompt_version]['desc']}")
else:
    selected_models_ab = st.sidebar.multiselect(
        "Pilih Model Target A/B Test",
        AVAILABLE_MODELS,
        default=["Gemini 3.6", "GPT-4o"],
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color: #8b9bb4; font-size: 0.85em; line-height: 1.5;'>"
    "👤 <b>Author:</b> <a href='https://github.com/suryapamungkas' target='_blank' style='color: #FF00FF; text-decoration: none;'>Nur Hidayat Surya Pamungkas</a><br>"
    "⚡ Powered by FastAPI + Gemini AI Judge + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<h1 class='glow-title'>✨ NEURAL EVAL CORE // DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Platform evaluasi, benchmarking, dan prompt engineering LLM dengan AI-as-a-Judge.</div>", unsafe_allow_html=True)


# ===========================================================================
# MODE 1: MODEL COMPARISON
# ===========================================================================
if app_mode == "🧪 Model Comparison":
    st.markdown("### 📝 INPUT PROMPT & EVALUATION CRITERIA")

    with st.form("eval_form"):
        st.markdown(
            f"<div style='background-color:rgba(138,43,226,0.15); padding:8px 14px; border-radius:6px; margin-bottom:12px; border:1px solid #8A2BE2;'>"
            f"<b>⚙️ Prompt Engineering Template Terpilih:</b> <code>{VERSION_MAP[prompt_version]['title']}</code><br>"
            f"<span style='color:#b5c4d8; font-size:0.88em;'>{VERSION_MAP[prompt_version]['desc']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        user_prompt = st.text_area(
            "Masukkan Query / Prompt",
            placeholder='Contoh: "Jelaskan konsep microservices architecture kepada developer pemula beserta kelebihan dan kekurangannya."',
            height=100,
        )

        with st.expander("🎯 Tambahkan Ground Truth / Referensi Kunci Jawaban (Opsional)"):
            ground_truth = st.text_area(
                "Ground Truth (Jawaban Ideal):",
                placeholder="Jawaban ideal harus memuat: dekomposisi layanan, database per service, komunikasi via API/gRPC, kelebihan independensi rilis, kekurangan kompleksitas jaringan...",
                height=80,
            )

        submitted = st.form_submit_button("🚀 INITIALIZE EVALUATION", use_container_width=True)

    if submitted and user_prompt.strip():
        if not selected_models:
            st.warning("Pilih minimal satu model di sidebar!")
        else:
            with st.spinner("⏳ Menjalankan evaluasi paralel dan AI Judge..."):
                try:
                    gt_val = ground_truth.strip() if ground_truth.strip() else None
                    results = _post_evaluate(
                        user_prompt.strip(),
                        selected_models,
                        prompt_version,
                        gt_val,
                    )
                except httpx.HTTPError as exc:
                    st.error(f"Gagal menghubungi backend: {exc}")
                    st.stop()

            st.session_state["last_results"] = results
            st.session_state["last_prompt"] = user_prompt.strip()
            st.session_state["last_gt"] = ground_truth.strip() if ground_truth.strip() else None

    # Results Section
    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        prompt_display = st.session_state["last_prompt"]
        gt_display = st.session_state.get("last_gt")

        st.divider()
        st.markdown("### 📊 SYSTEM TELEMETRY: KOMPARASI MODEL")
        st.markdown(f"**Query:** <span style='font-family: monospace; color:#FF00FF'>{prompt_display}</span>", unsafe_allow_html=True)
        if gt_display:
            st.markdown(f"**Ground Truth:** <span style='font-family: monospace; color:#39FF14'>{gt_display}</span>", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        df["avg_score"] = (
            (df["correctness_score"] + df["relevance_score"] + df["faithfulness_score"]) / 3
        ).round(4)

        min_lat = df["latency_seconds"].min()
        df["status"] = df.apply(lambda r: _status_badge(r, min_lat), axis=1)

        # Telemetry Table
        summary = df[["model_name", "avg_score", "latency_seconds", "estimated_cost_usd", "total_tokens", "status"]].copy()
        summary.columns = ["Model", "Score (Avg)", "Latency (s)", "Cost (USD)", "Tokens", "Status"]
        st.dataframe(
            summary.style.format(
                {"Score (Avg)": "{:.4f}", "Latency (s)": "{:.3f}s", "Cost (USD)": "${:.6f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Charts
        fig_scores = px.bar(
            df,
            x="model_name",
            y=["correctness_score", "relevance_score", "faithfulness_score"],
            barmode="group",
            title="Perbandingan Skor Kualitas per Model (Correctness, Relevance, Faithfulness)",
            labels={"value": "Score (0-1)", "model_name": "Model", "variable": "Metrik"},
            color_discrete_sequence=["#8A2BE2", "#FF00FF", "#39FF14"],
        )
        fig_scores.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scores, use_container_width=True)

        col_lat, col_cost = st.columns(2)
        with col_lat:
            fig_lat = px.bar(
                df,
                x="model_name",
                y="latency_seconds",
                title="Latensi Waktu Respons (detik)",
                labels={"latency_seconds": "Detik", "model_name": "Model"},
                color="model_name",
                color_discrete_sequence=NEON_PALETTE,
            )
            fig_lat.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_cost:
            fig_cost = px.bar(
                df,
                x="model_name",
                y="estimated_cost_usd",
                title="Estimasi Biaya Eksekusi (USD)",
                labels={"estimated_cost_usd": "USD", "model_name": "Model"},
                color="model_name",
                color_discrete_sequence=NEON_PALETTE,
            )
            fig_cost.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cost, use_container_width=True)

        # -------------------------------------------------------------------
        # 🔍 Drill-Down View – Tabs & AI Judge Reasoning
        # -------------------------------------------------------------------
        st.divider()
        st.markdown("### 🔍 DETAIL RESPONS & AI JUDGE REASONING")

        if results:
            tabs = st.tabs([r["model_name"] for r in results])
            for idx, tab in enumerate(tabs):
                with tab:
                    record = results[idx]
                    version_tag = record.get("prompt_version", "v1")
                    version_info = VERSION_MAP.get(version_tag, {"title": version_tag, "badge": version_tag, "desc": ""})

                    st.markdown(f"**🏷️ Versi Prompt yang Dieksekusi:** `{version_info['badge']}` — *{version_info['desc']}*")

                    st.text_area(
                        f"Output ({record['model_name']})",
                        value=record["output_text"],
                        height=180,
                        key=f"output_{record['id']}",
                        disabled=True,
                    )

                    # AI Judge Reasoning Box
                    reasoning_text = record.get("judge_reasoning") or "Evaluasi dihitung berbasis benchmark."
                    st.markdown(
                        f"<div class='judge-box'>🧠 <b>AI Judge Analysis & Reasoning:</b><br>{reasoning_text}</div>",
                        unsafe_allow_html=True,
                    )

                    m_col1, m_col2 = st.columns([1, 1])
                    with m_col1:
                        st.markdown("**Breakdown Skor Kualitas:**")
                        score_data = {
                            "Metrik": ["Correctness", "Relevance", "Faithfulness"],
                            "Skor": [
                                record["correctness_score"],
                                record["relevance_score"],
                                record["faithfulness_score"],
                            ],
                        }
                        st.dataframe(
                            pd.DataFrame(score_data).style.format({"Skor": "{:.4f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with m_col2:
                        st.markdown("**Statistik Request:**")
                        st.metric("Latency", f"{record['latency_seconds']:.3f}s")
                        st.metric("Tokens", record["total_tokens"])
                        st.metric("Cost", f"${record['estimated_cost_usd']:.6f}")


# ===========================================================================
# MODE 2: PROMPT A/B TESTING
# ===========================================================================
else:
    st.markdown("### ⚖️ PROMPT ENGINEERING A/B LAB")
    st.markdown("Uji dan bandingkan performa dua variasi prompt pada model LLM target.")

    with st.form("ab_form"):
        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.markdown("#### 🅰️ Prompt Versi A (Baseline)")
            prompt_a = st.text_area(
                "Prompt A:",
                value="Jelaskan Docker container secara singkat.",
                height=120,
            )
        with col_pb:
            st.markdown("#### 🅱️ Prompt Versi B (Candidate / Refined)")
            prompt_b = st.text_area(
                "Prompt B:",
                value="Sebagai Senior DevOps Engineer, jelaskan Docker container dengan analogi pengapalan logistik. Sertakan 3 manfaat utamanya secara ringkas dan terstruktur.",
                height=120,
            )

        with st.expander("🎯 Ground Truth / Target Jawaban (Opsional untuk A/B Test)"):
            ground_truth_ab = st.text_area(
                "Kunci Jawaban Referensi:",
                placeholder="Jawaban ideal menjelaskan isolasi aplikasi, portabilitas lintas OS, dan efisiensi resource dibanding VM.",
                height=70,
            )

        submitted_ab = st.form_submit_button("⚖️ RUN A/B COMPARISON TEST", use_container_width=True)

    if submitted_ab and prompt_a.strip() and prompt_b.strip():
        if not selected_models_ab:
            st.warning("Pilih minimal satu model target di sidebar!")
        else:
            with st.spinner("⏳ Menjalankan A/B Testing paralel untuk Prompt A dan Prompt B..."):
                try:
                    gt_ab_val = ground_truth_ab.strip() if ground_truth_ab.strip() else None
                    res_a = _post_evaluate(prompt_a.strip(), selected_models_ab, "Prompt A", gt_ab_val)
                    res_b = _post_evaluate(prompt_b.strip(), selected_models_ab, "Prompt B", gt_ab_val)
                except httpx.HTTPError as exc:
                    st.error(f"Gagal menghubungi backend: {exc}")
                    st.stop()

            st.session_state["ab_results_a"] = res_a
            st.session_state["ab_results_b"] = res_b
            st.session_state["ab_prompt_a"] = prompt_a.strip()
            st.session_state["ab_prompt_b"] = prompt_b.strip()

    # Display A/B Test Results
    if "ab_results_a" in st.session_state and "ab_results_b" in st.session_state:
        res_a = st.session_state["ab_results_a"]
        res_b = st.session_state["ab_results_b"]

        df_a = pd.DataFrame(res_a)
        df_b = pd.DataFrame(res_b)

        df_a["avg_score"] = ((df_a["correctness_score"] + df_a["relevance_score"] + df_a["faithfulness_score"]) / 3).round(4)
        df_b["avg_score"] = ((df_b["correctness_score"] + df_b["relevance_score"] + df_b["faithfulness_score"]) / 3).round(4)

        avg_score_a = df_a["avg_score"].mean()
        avg_score_b = df_b["avg_score"].mean()
        avg_lat_a = df_a["latency_seconds"].mean()
        avg_lat_b = df_b["latency_seconds"].mean()
        avg_cost_a = df_a["estimated_cost_usd"].mean()
        avg_cost_b = df_b["estimated_cost_usd"].mean()

        delta_score = avg_score_b - avg_score_a
        delta_lat = avg_lat_b - avg_lat_a
        delta_cost = avg_cost_b - avg_cost_a

        st.divider()

        # Winner Announcement Card
        if delta_score > 0:
            winner_text = f"🏆 **Prompt B Menang!** Peningkatan kualitas rata-rata sebesar **+{delta_score * 100:.1f}%**"
        elif delta_score < 0:
            winner_text = f"🏆 **Prompt A Lebih Unggul!** Prompt A memiliki skor lebih tinggi sebesar **+{abs(delta_score) * 100:.1f}%**"
        else:
            winner_text = "⚖️ **Hasil Seimbang!** Kedua prompt menghasilkan kualitas yang setara."

        st.markdown(f"<div class='winner-card'><h3>{winner_text}</h3></div>", unsafe_allow_html=True)

        # Delta Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Skor Kualitas (Avg B vs A)", f"{avg_score_b:.4f}", delta=f"{delta_score:+.4f}")
        m2.metric("Latensi (Avg B vs A)", f"{avg_lat_b:.3f}s", delta=f"{delta_lat:+.3f}s", delta_color="inverse")
        m3.metric("Biaya (Avg B vs A)", f"${avg_cost_b:.6f}", delta=f"${delta_cost:+.6f}", delta_color="inverse")
        m4.metric("Jumlah Model Diuji", len(res_a))

        # Side-by-side Response Inspection
        st.markdown("### 🔍 INSPEKSI RESPONS & JUDGE PER MODEL")
        for idx in range(len(res_a)):
            item_a = res_a[idx]
            item_b = res_b[idx]
            model_name = item_a["model_name"]

            with st.expander(f"📌 Model: {model_name} (Klik untuk expand detail A vs B)", expanded=True):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**🅰️ Output Prompt A (Skor: {item_a['correctness_score']:.2f} / {item_a['relevance_score']:.2f} / {item_a['faithfulness_score']:.2f})**")
                    st.text_area("Output A", value=item_a["output_text"], height=160, key=f"ab_out_a_{idx}", disabled=True)
                    st.markdown(f"<div class='judge-box'>🧠 <b>Judge A:</b> {item_a.get('judge_reasoning', 'N/A')}</div>", unsafe_allow_html=True)
                    st.caption(f"⏱️ Latency: {item_a['latency_seconds']:.3f}s | 🪙 Tokens: {item_a['total_tokens']} | 💰 Cost: ${item_a['estimated_cost_usd']:.6f}")

                with cb:
                    st.markdown(f"**🅱️ Output Prompt B (Skor: {item_b['correctness_score']:.2f} / {item_b['relevance_score']:.2f} / {item_b['faithfulness_score']:.2f})**")
                    st.text_area("Output B", value=item_b["output_text"], height=160, key=f"ab_out_b_{idx}", disabled=True)
                    st.markdown(f"<div class='judge-box'>🧠 <b>Judge B:</b> {item_b.get('judge_reasoning', 'N/A')}</div>", unsafe_allow_html=True)
                    st.caption(f"⏱️ Latency: {item_b['latency_seconds']:.3f}s | 🪙 Tokens: {item_b['total_tokens']} | 💰 Cost: ${item_b['estimated_cost_usd']:.6f}")

        # Comparison Bar Chart
        st.markdown("### 📊 GRAFIK KOMPARASI METRIK A vs B")
        comp_df = pd.DataFrame([
            {"Model": r["model_name"], "Prompt": "Prompt A", "Score": (r["correctness_score"] + r["relevance_score"] + r["faithfulness_score"]) / 3, "Latency": r["latency_seconds"], "Cost": r["estimated_cost_usd"]}
            for r in res_a
        ] + [
            {"Model": r["model_name"], "Prompt": "Prompt B", "Score": (r["correctness_score"] + r["relevance_score"] + r["faithfulness_score"]) / 3, "Latency": r["latency_seconds"], "Cost": r["estimated_cost_usd"]}
            for r in res_b
        ])

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            fig_ab_score = px.bar(
                comp_df,
                x="Model",
                y="Score",
                color="Prompt",
                barmode="group",
                title="Perbandingan Skor Kualitas (A vs B)",
                color_discrete_sequence=["#8A2BE2", "#FF00FF"],
            )
            fig_ab_score.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ab_score, use_container_width=True)

        with chart_c2:
            fig_ab_lat = px.bar(
                comp_df,
                x="Model",
                y="Latency",
                color="Prompt",
                barmode="group",
                title="Perbandingan Latensi (detik)",
                color_discrete_sequence=["#8A2BE2", "#FF00FF"],
            )
            fig_ab_lat.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ab_lat, use_container_width=True)


# ===========================================================================
# 📈 Historical Analytics & Export (Shared across modes)
# ===========================================================================
st.divider()
st.markdown("### 📈 RIWAYAT EVALUASI & EXPORT")
try:
    history = _get_results(limit=200)
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist["avg_score"] = (
            (df_hist["correctness_score"] + df_hist["relevance_score"] + df_hist["faithfulness_score"]) / 3
        ).round(4)

        # CSV Download Button
        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Unduh Seluruh Riwayat Evaluasi (CSV)",
            data=csv_data,
            file_name="llm_evaluation_history.csv",
            mime="text/csv",
        )

        fig_hist = px.line(
            df_hist.sort_values("timestamp"),
            x="timestamp",
            y="avg_score",
            color="model_name",
            title="Tren Skor Kualitas Seiring Waktu",
            labels={"avg_score": "Avg Score", "timestamp": "Waktu", "model_name": "Model"},
            markers=True,
            color_discrete_sequence=NEON_PALETTE,
        )
        fig_hist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

        with st.expander("📋 Tabel Riwayat Lengkap Database"):
            st.dataframe(
                df_hist[
                    [
                        "timestamp", "model_name", "prompt_version", "input_prompt",
                        "avg_score", "latency_seconds", "estimated_cost_usd", "total_tokens",
                    ]
                ].sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Belum ada data evaluasi. Jalankan evaluasi di atas untuk memulai!")
except httpx.HTTPError:
    st.info("Backend belum aktif atau tidak dapat dijangkau. Jalankan backend terlebih dahulu.")

# ---------------------------------------------------------------------------
# Footer – Author & Copyright Branding
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='text-align: center; color: #8b9bb4; padding: 25px 0 15px 0; font-family: monospace; font-size: 0.88em; border-top: 1px solid rgba(138, 43, 226, 0.25); margin-top: 50px;'>"
    "✨ <b>Neural Eval Core</b> • Created & Maintained by "
    "<a href='https://github.com/suryapamungkas' target='_blank' style='color: #FF00FF; font-weight: bold; text-decoration: none;'>Nur Hidayat Surya Pamungkas</a>"
    "</div>",
    unsafe_allow_html=True,
)
