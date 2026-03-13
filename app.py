"""
app.py — WhyML: Interactive Model Explanation Tool
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
from model.explain import (
    plot_waterfall, plot_beeswarm, plot_bar_importance,
    predict_proba, counterfactual_explanation
)

MODEL_DIR = os.path.join(ROOT, 'model')
DATA_DIR  = os.path.join(ROOT, 'data')

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhyML — Explainable AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1D9E75;
        margin-bottom: 0; padding-bottom: 0;
    }
    .sub-header { color: #666; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 16px 20px;
        border-left: 4px solid #1D9E75; margin-bottom: 8px;
    }
    .risk-high {
        background: #fff0ee; border-radius: 10px; padding: 16px;
        border-left: 4px solid #D85A30; text-align: center;
    }
    .risk-low {
        background: #e8f8f2; border-radius: 10px; padding: 16px;
        border-left: 4px solid #1D9E75; text-align: center;
    }
    .section-title {
        font-size: 1.2rem; font-weight: 600;
        border-bottom: 2px solid #1D9E75; padding-bottom: 6px; margin-top: 20px;
    }
    .cf-card {
        background: #fffbe6; border-radius: 8px; padding: 12px 16px;
        border: 1px solid #f0d060; margin-bottom: 8px; font-size: 0.92rem;
    }
    .cf-flip {
        background: #e8f8f2; border: 1px solid #1D9E75;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    meta      = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl'))
    test_data = joblib.load(os.path.join(MODEL_DIR, 'test_data.pkl'))
    df        = pd.read_csv(os.path.join(DATA_DIR, 'heart_disease.csv'))
    X_bg      = df.drop('target', axis=1).sample(100, random_state=42)
    return meta, test_data, df, X_bg


# ── check models are trained ──────────────────────────────────────────────────
def models_exist():
    return os.path.exists(os.path.join(MODEL_DIR, 'random_forest.pkl'))


if not models_exist():
    st.warning("⚙️ First run — training models... (this takes ~15 seconds)")
    with st.spinner("Training Random Forest, XGBoost, and Gradient Boosting..."):
        import subprocess
        subprocess.run([sys.executable, os.path.join(MODEL_DIR, 'train.py')], check=True)
    st.success("Models trained! Refreshing...")
    st.rerun()

meta, test_data, df, X_bg = load_assets()
feature_info = meta['feature_info']
feature_names = meta['feature_names']
X_test = test_data['X_test']
y_test = test_data['y_test']


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model_choice = st.selectbox(
        "Select Model",
        ['random_forest', 'xgboost', 'gradient_boosting'],
        format_func=lambda x: {
            'random_forest': '🌲 Random Forest',
            'xgboost': '⚡ XGBoost',
            'gradient_boosting': '🔥 Gradient Boosting'
        }[x]
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    results = meta['results']
    for name, r in results.items():
        label = {'random_forest': 'Random Forest',
                 'xgboost': 'XGBoost',
                 'gradient_boosting': 'Grad. Boosting'}[name]
        active = "✅ " if name == model_choice else ""
        st.markdown(f"**{active}{label}**")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{r['accuracy']:.1%}")
        col2.metric("AUC", f"{r['auc']:.3f}")

    st.markdown("---")
    st.markdown("### 🎲 Quick Load")
    if st.button("Load Random Test Case"):
        idx = np.random.choice(X_test.index)
        st.session_state['loaded_idx'] = idx
    if st.button("Load High-Risk Case"):
        high_risk = y_test[y_test == 1].index
        if len(high_risk):
            st.session_state['loaded_idx'] = np.random.choice(high_risk)
    if st.button("Load Low-Risk Case"):
        low_risk = y_test[y_test == 0].index
        if len(low_risk):
            st.session_state['loaded_idx'] = np.random.choice(low_risk)

    st.markdown("---")
    st.markdown(
        "<small>**WhyML** — Interpretable AI Dashboard<br>"
        "Built with SHAP + Streamlit</small>",
        unsafe_allow_html=True
    )


# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 WhyML</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Explainable AI — understand <em>why</em> your model makes predictions</p>',
    unsafe_allow_html=True
)
st.markdown("---")

tabs = st.tabs(["🎯 Predict & Explain", "🌍 Global Insights", "⚖️ Compare Models", "📘 About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Predict & Explain
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title">Patient Input Features</p>', unsafe_allow_html=True)

    # prefill from sidebar quick-load
    defaults = {}
    if 'loaded_idx' in st.session_state:
        idx = st.session_state['loaded_idx']
        row = X_test.loc[idx]
        defaults = row.to_dict()

    # build input grid
    cols_per_row = 3
    feature_vals = {}
    all_features = list(feature_info.items())

    for row_i in range(0, len(all_features), cols_per_row):
        row_feats = all_features[row_i: row_i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (feat, info) in zip(cols, row_feats):
            with col:
                default_val = defaults.get(feat, info['default'])
                if info['step'] == 1:
                    val = st.number_input(
                        info['label'],
                        min_value=int(info['min']),
                        max_value=int(info['max']),
                        value=int(default_val),
                        step=1,
                        key=f"input_{feat}"
                    )
                else:
                    val = st.number_input(
                        info['label'],
                        min_value=float(info['min']),
                        max_value=float(info['max']),
                        value=float(default_val),
                        step=float(info['step']),
                        key=f"input_{feat}"
                    )
                feature_vals[feat] = val

    st.markdown("")
    predict_btn = st.button("🔍 Explain This Prediction", type="primary", use_container_width=True)

    if predict_btn or 'last_input' in st.session_state:
        input_df = pd.DataFrame([feature_vals])[feature_names]
        st.session_state['last_input'] = feature_vals

        prob_no, prob_yes = predict_proba(model_choice, input_df)
        prediction = 1 if prob_yes >= 0.5 else 0

        st.markdown("---")

        # ── prediction banner ─────────────────────────────────────────────
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2 style="color:#D85A30; margin:0;">⚠️ High Risk</h2>
                    <p style="font-size:2rem; font-weight:700; color:#D85A30; margin:4px 0;">
                        {prob_yes:.1%}
                    </p>
                    <p style="color:#666; margin:0;">probability of heart disease</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2 style="color:#1D9E75; margin:0;">✅ Low Risk</h2>
                    <p style="font-size:2rem; font-weight:700; color:#1D9E75; margin:4px 0;">
                        {prob_no:.1%}
                    </p>
                    <p style="color:#666; margin:0;">probability of being healthy</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── gauge chart ───────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_yes * 100, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk %", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar':  {'color': "#D85A30" if prob_yes > 0.5 else "#1D9E75"},
                'steps': [
                    {'range': [0, 30],  'color': '#e8f8f2'},
                    {'range': [30, 60], 'color': '#fff8e6'},
                    {'range': [60, 100],'color': '#fff0ee'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP waterfall ────────────────────────────────────────────────
        st.markdown('<p class="section-title">🔬 Why This Prediction? (SHAP Waterfall)</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each bar shows how much a feature **pushed the prediction higher (red) or lower (blue)** "
            "from the model's baseline."
        )
        with st.spinner("Computing SHAP values..."):
            fig_wf = plot_waterfall(model_choice, input_df, X_bg)
        st.pyplot(fig_wf, use_container_width=True)
        plt.close('all')

        # ── counterfactual ────────────────────────────────────────────────
        st.markdown('<p class="section-title">💡 Counterfactual Explanations</p>',
                    unsafe_allow_html=True)
        outcome = "High Risk" if prediction == 1 else "Low Risk"
        flip_to = "Low Risk" if prediction == 1 else "High Risk"
        st.markdown(
            f"Current prediction: **{outcome}**. "
            f"Here's what could change it towards **{flip_to}**:"
        )

        with st.spinner("Computing counterfactuals..."):
            cfs = counterfactual_explanation(model_choice, input_df, X_bg, feature_info)

        if cfs:
            for cf in cfs:
                card_class = "cf-card cf-flip" if cf['flips'] else "cf-card"
                arrow = "⬆️" if cf['direction'] == "increase" else "⬇️"
                flip_badge = "🔄 <strong>Flips prediction!</strong>" if cf['flips'] else ""
                shift_color = "#1D9E75" if cf['prob_shift'] < 0 else "#D85A30"
                st.markdown(f"""
                <div class="{card_class}">
                    {arrow} <strong>{cf['label']}</strong>:
                    {cf['current']} → <strong>{cf['suggested']}</strong>
                    &nbsp;|&nbsp;
                    <span style="color:{shift_color}">
                        risk shift: {cf['prob_shift']:+.1%}
                    </span>
                    &nbsp; {flip_badge}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No strong counterfactuals found — prediction is robust to feature changes.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Global Insights
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title">🌍 Global Feature Importance</p>',
                unsafe_allow_html=True)
    st.markdown("Averaged across all test samples — which features matter most overall?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean |SHAP| Bar Chart**")
        with st.spinner("Computing global importance..."):
            fig_bar = plot_bar_importance(model_choice, X_bg)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close('all')

    with col2:
        st.markdown("**SHAP Beeswarm (Distribution)**")
        st.markdown(
            "<small style='color:#888'>Each dot = one patient. "
            "Red = high feature value, Blue = low. "
            "Position shows SHAP impact.</small>",
            unsafe_allow_html=True
        )
        with st.spinner("Computing beeswarm..."):
            fig_bee = plot_beeswarm(model_choice, X_bg)
        st.pyplot(fig_bee, use_container_width=True)
        plt.close('all')

    # ── dataset stats ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Dataset Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df))
    c2.metric("Features", len(feature_names))
    c3.metric("Heart Disease +", int(df['target'].sum()))
    c4.metric("Healthy", int((df['target'] == 0).sum()))

    st.markdown("**Feature Distributions**")
    feat_to_plot = st.selectbox("Select feature", feature_names, key="dist_feat")
    fig_hist, ax = plt.subplots(figsize=(8, 3))
    for label, color in [(0, '#1D9E75'), (1, '#D85A30')]:
        subset = df[df['target'] == label][feat_to_plot]
        ax.hist(subset, bins=20, alpha=0.6, color=color,
                label='Healthy' if label == 0 else 'Heart Disease')
    ax.set_xlabel(feature_info.get(feat_to_plot, {}).get('label', feat_to_plot))
    ax.set_ylabel('Count')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    fig_hist.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig_hist, use_container_width=True)
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Compare Models
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">⚖️ Side-by-Side Model Comparison</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Same input, different models — do they agree? Do they explain differently?"
    )

    if 'last_input' not in st.session_state:
        st.info("👆 Go to **Predict & Explain**, enter values and click Explain first.")
    else:
        input_df_cmp = pd.DataFrame([st.session_state['last_input']])[feature_names]

        model_names = ['random_forest', 'xgboost', 'gradient_boosting']
        model_labels = {
            'random_forest': '🌲 Random Forest',
            'xgboost': '⚡ XGBoost',
            'gradient_boosting': '🔥 Gradient Boosting'
        }

        probs = {}
        for m in model_names:
            p0, p1 = predict_proba(m, input_df_cmp)
            probs[m] = p1

        # agreement indicator
        preds = [round(p) for p in probs.values()]
        agreement = len(set(preds)) == 1
        if agreement:
            st.success(f"✅ All 3 models **agree**: {'High Risk' if preds[0]==1 else 'Low Risk'}")
        else:
            st.warning("⚠️ Models **disagree** on this prediction — review carefully.")

        # probability bars
        fig_cmp = go.Figure()
        for m in model_names:
            p = probs[m]
            color = "#D85A30" if p > 0.5 else "#1D9E75"
            fig_cmp.add_trace(go.Bar(
                name=model_labels[m],
                x=[model_labels[m]],
                y=[round(p * 100, 1)],
                marker_color=color,
                text=[f"{p:.1%}"],
                textposition='outside'
            ))
        fig_cmp.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="Decision boundary (50%)")
        fig_cmp.update_layout(
            title="Risk Probability by Model",
            yaxis_title="Risk Probability (%)",
            yaxis_range=[0, 110],
            showlegend=False,
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # SHAP waterfall comparison
        st.markdown("**SHAP Explanation Comparison**")
        cols = st.columns(3)
        for col, m in zip(cols, model_names):
            with col:
                st.markdown(f"**{model_labels[m]}**")
                with st.spinner(f"Computing {m}..."):
                    fig = plot_waterfall(m, input_df_cmp, X_bg)
                st.pyplot(fig, use_container_width=True)
                plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: About
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    ## 📘 About WhyML

    **WhyML** is an end-to-end Explainable AI (XAI) dashboard demonstrating how to make
    machine learning models interpretable and trustworthy.

    ### What this project covers
    - **SHAP (SHapley Additive exPlanations)** — a game-theoretic approach to explain individual predictions
    - **Waterfall plots** — visualize how each feature contributes to a single prediction
    - **Beeswarm plots** — show global feature importance distribution across all samples
    - **Counterfactual explanations** — "what would need to change to flip this prediction?"
    - **Model comparison** — do different models agree? Do they explain differently?

    ### Why Explainability Matters
    In high-stakes domains like **healthcare, finance, and law**, knowing *why* a model
    predicted something is as important as the prediction itself. XAI bridges the gap between
    black-box accuracy and human trust.

    ### Technical Stack
    | Component | Technology |
    |---|---|
    | Explainability | SHAP (TreeExplainer) |
    | Models | Random Forest, XGBoost, Gradient Boosting |
    | Frontend | Streamlit |
    | Visualization | Matplotlib, Plotly |
    | Dataset | Synthetic Heart Disease (clinically derived features) |

    ### Key Concepts
    - **SHAP value**: how much a feature pushed the prediction away from the baseline
    - **Base value**: average model output across training data
    - **Counterfactual**: the minimal change to flip a prediction
    - **TreeExplainer**: exact SHAP computation for tree-based models (fast & exact)

    ### References
    - Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*
    - Wachter et al. (2018) — *Counterfactual Explanations Without Opening the Black Box*
    - Molnar (2022) — *Interpretable Machine Learning* (free book)

    ---
    Built as a capstone project in Explainable AI (XAI) · SHAP + Streamlit · Python 3.11
    """)
