"""Streamlit app for patient-modulated CCM-RWI analysis."""

import streamlit as st
from engine import (
    DEFAULTS,
    build_decision_cache,
    compute_patient_risks,
    make_single_strata_figure,
)


st.set_page_config(
    page_title="CCM-RWI Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    .stApp { background-color: #F7F9FC; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Primary button — blue */
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        border: 1px solid #2563EB !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:focus {
        background-color: #1E40AF !important;
        border-color: #1E40AF !important;
        color: white !important;
    }
    /* Secondary button — blue outlined */
    .stButton > button[kind="secondary"] {
        background-color: white !important;
        border: 1px solid #2563EB !important;
        color: #2563EB !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[kind="secondary"]:focus {
        background-color: #EFF6FF !important;
        border-color: #1E40AF !important;
        color: #1E40AF !important;
    }

    /* Slider — round blue thumb, blue filled track, value visible */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2563EB !important;
        border-radius: 50% !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {
        background-color: #2563EB !important;
    }

    /* Selected radio button */
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
    }


    .section-title {
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-size: 1.15em;
        color: #1A202C;
        font-weight: 700;
        margin: 1.5em 0 0.6em 0;
        padding-bottom: 0.4em;
        border-bottom: 1px solid #DDE3EC;
    }
    .small-help {
        color: #4A5568;
        font-size: 0.85em;
        margin: 0 0 1em 0;
    }

    .result-card {
        background: white;
        border-radius: 8px;
        padding: 1.4em 1.6em;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid #DDE3EC;
        margin-bottom: 0.8em;
    }
    .result-card.observation { border-top: 4px solid #2563EB; }
    .result-card.resection { border-top: 4px solid #DC2626; }
    .result-card .arm-name {
        font-size: 1.05em; color: #1A202C; font-weight: 700; margin: 0 0 1em 0;
    }
    .result-label {
        font-size: 0.85em; color: #4A5568; font-weight: 500; margin: 0 0 0.2em 0;
    }
    .result-value {
        font-size: 1.6em; color: #1A202C; font-weight: 700;
        font-family: ui-monospace, "Menlo", monospace; margin: 0 0 1em 0;
    }
    .result-value:last-child { margin-bottom: 0; }

    .recommendation-banner {
        background: white; border-radius: 8px;
        padding: 1.2em 1.6em; margin: 1em 0;
        border: 1px solid #DDE3EC;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        color: #1A202C; font-size: 1em; line-height: 1.7;
    }
    .recommendation-banner strong { color: #1A202C; }
    .recommend-resection { color: #DC2626; font-weight: 700; }
    .recommend-observation { color: #2563EB; font-weight: 700; }

    .stRadio > label { font-weight: 600; color: #1A202C; }
    .stSlider > label { font-weight: 500; color: #1A202C; font-size: 0.92em; }
</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align: center; margin: 0.5em 0;'>"
    "Reframing Cerebral Cavernous Malformation Management:<br>"
    "A Risk-Weighted Impact Analysis of Natural History versus Surgical Intervention"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4A5568; font-size: 0.95em; "
    "margin: 0.8em 0 1.5em 0; line-height: 1.5;'>"
    "Interactive calculator for the cumulative lifetime poor-outcome risk and Risk-Weighted "
    "Impact of observation versus microsurgical resection in cerebral cavernous malformations. "
    "Adjust the input parameters below to reflect the specific clinical scenario."
    "</p>",
    unsafe_allow_html=True,
)




# ─── Patient characteristics ────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient characteristics</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    location_label = st.radio("Lesion location",
                              ["Non-brainstem", "Brainstem"],
                              horizontal=True, key="loc_radio")
    location = "brainstem" if location_label == "Brainstem" else "nonbrainstem"
with col2:
    sex_label = st.radio("Sex", ["Female", "Male"], horizontal=True, key="sex_radio")
    sex = sex_label.lower()
with col3:
    presentation = st.radio("Presentation",
                            ["Asymptomatic", "Symptomatic"],
                            horizontal=True, key="pres_radio")
    is_sympt = presentation == "Symptomatic"


# ─── Age and years-since-bleed (top, under patient characteristics) ─────────
col_age, col_ysb = st.columns(2)
with col_age:
    age = st.slider("Age at decision (years)", 20, 80, 40, key="age")
with col_ysb:
    if is_sympt:
        years_since_bleed = st.slider(
            "Years since first symptomatic event",
            min_value=0, max_value=10,
            value=0, step=1,
            key="years_since_bleed",
        )
        presentation_state = min(years_since_bleed + 1, DEFAULTS["years_elevated"])
    else:
        presentation_state = 0


# ─── Model parameters ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Model parameters</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="small-help">'
    'Default values are population averages from the published literature. '
    'Adjust the sliders if the patient or clinical context differs from these defaults '
    '(e.g., surgeon-specific complication rates, lesion-specific risk estimates).'
    '</p>',
    unsafe_allow_html=True,
)


nh_asympt = DEFAULTS["nh"][(location, False)]
nh_sympt = DEFAULTS["nh"][(location, True)]
tx_default = DEFAULTS["tx"][("resection", location)]
default_baseline_util = (DEFAULTS["util_well_bs"] if location == "brainstem"
                          else DEFAULTS["util_well_nbs"])


col1, col2 = st.columns(2)
with col1:
    st.markdown("**Natural-history parameters**")
    nh_baseline_rate = st.slider(
        "Asymptomatic baseline annual hemorrhage rate (%/yr)",
        min_value=0.0, max_value=10.0,
        value=float(nh_asympt["late"] * 100),
        step=0.01, format="%.2f",
        key=f"nh_baseline_{location}",
    ) / 100

    if is_sympt:
        nh_elevated_rate = st.slider(
            "Symptomatic elevated annual hemorrhage rate (%/yr)",
            min_value=0.0, max_value=20.0,
            value=float(nh_sympt["early"] * 100),
            step=0.01, format="%.2f",
            key=f"nh_elevated_{location}",
        ) / 100
    else:
        nh_elevated_rate = nh_sympt["early"]

    years_elevated = st.slider(
        "Years of elevated hemorrhage risk after a bleed",
        min_value=1, max_value=10,
        value=DEFAULTS["years_elevated"], step=1,
        key="years_elevated",
    )

    p_death = st.slider(
        "Mortality per hemorrhage (%)",
        min_value=0.0, max_value=20.0,
        value=float(nh_asympt["p_death"] * 100),
        step=0.1, format="%.1f",
        key=f"p_death_{location}",
    ) / 100

    p_disab = st.slider(
        "Severe disability (mRS 3–5) per hemorrhage (%)",
        min_value=0.0, max_value=50.0,
        value=float(nh_asympt["p_disab"] * 100),
        step=0.1, format="%.1f",
        key=f"p_disab_{location}",
    ) / 100

with col2:
    st.markdown("**Treatment parameters**")
    tx_p_death = st.slider(
        "Surgical mortality (%)",
        min_value=0.0, max_value=10.0,
        value=float(tx_default["p_death"] * 100),
        step=0.1, format="%.1f",
        key=f"tx_p_death_{location}",
    ) / 100

    tx_p_disab = st.slider(
        "Surgical severe disability (mRS 3–5) (%)",
        min_value=0.0, max_value=50.0,
        value=float(tx_default["p_disab"] * 100),
        step=0.1, format="%.1f",
        key=f"tx_p_disab_{location}",
    ) / 100

    baseline_util = st.slider(
        f"Utility of well with {location_label.lower()} CCM",
        min_value=0.50, max_value=1.00,
        value=float(default_baseline_util),
        step=0.01, format="%.2f",
        key=f"baseline_util_{location}",
    )

    disabled_util = st.slider(
        "Utility of severe disability (mRS 3–5)",
        min_value=0.0, max_value=baseline_util,
        value=float(DEFAULTS["util_disabled"]),
        step=0.01, format="%.2f",
        key="disabled_util",
    )

    # Baseline-referenced disability severity weight, derived from the two
    # utilities so a disabled life-year loses exactly (baseline − disabled),
    # sharing death's reference point. Matches paper_release/ccm_rwi.py.
    weight_disability = (
        (baseline_util - disabled_util) / baseline_util if baseline_util > 0 else 0.0
    )


# ─── Calculate ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
calculate = st.button("Calculate", type="primary", use_container_width=True)


# ─── Results ────────────────────────────────────────────────────────────────
if calculate:
    obs_rwi, obs_cum, res_rwi, res_cum = compute_patient_risks(
        age, sex, location, presentation_state,
        nh_baseline_rate, nh_elevated_rate,
        p_death, p_disab,
        tx_p_death, tx_p_disab,
        weight_disability, baseline_util, years_elevated,
    )

    st.markdown('<div class="section-title">Projected outcomes</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="result-card observation">
            <div class="arm-name">Observation</div>
            <div class="result-label">Lifetime poor-outcome risk</div>
            <div class="result-value">{obs_cum*100:.1f}%</div>
            <div class="result-label">Risk-Weighted Impact</div>
            <div class="result-value">{obs_rwi:.2f} yr</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="result-card resection">
            <div class="arm-name">Microsurgical resection</div>
            <div class="result-label">Lifetime poor-outcome risk</div>
            <div class="result-value">{res_cum*100:.1f}%</div>
            <div class="result-label">Risk-Weighted Impact</div>
            <div class="result-value">{res_rwi:.2f} yr</div>
        </div>
        """, unsafe_allow_html=True)

    d_cum = obs_cum - res_cum
    d_rwi = obs_rwi - res_rwi
    cum_rec = "Resection" if d_cum > 0 else "Observation"
    rwi_rec = "Resection" if d_rwi > 0 else "Observation"
    cum_cls = "recommend-resection" if d_cum > 0 else "recommend-observation"
    rwi_cls = "recommend-resection" if d_rwi > 0 else "recommend-observation"

    st.markdown(f"""
    <div class="recommendation-banner">
        <strong>Cumulative-risk policy:</strong>
        <span class="{cum_cls}">{cum_rec}</span>
        &nbsp; (Δ = {d_cum*100:+.2f} percentage points)<br>
        <strong>RWI policy:</strong>
        <span class="{rwi_cls}">{rwi_rec}</span>
        &nbsp; (Δ = {d_rwi:+.3f} years)
    </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">Decision-zone map</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="small-help">'
        'Decision-zone recommendations across age for this patient\'s exact stratum, under the '
        'parameters above. Solid vertical line marks the cumulative-risk crossover age; '
        'dashed line marks the RWI crossover age. The patient\'s age is outlined in red.'
        '</p>',
        unsafe_allow_html=True,
    )

    user_overrides = {
        "nh_baseline_rate": nh_baseline_rate,
        "nh_elevated_rate": nh_elevated_rate,
        "p_death_given_bleed": p_death,
        "p_disability_given_bleed": p_disab,
        "tx_p_death": tx_p_death,
        "tx_p_disab": tx_p_disab,
    }
    cache = build_decision_cache(
        user_location=location,
        user_overrides=user_overrides,
        weight_disability=weight_disability,
        years_elevated=years_elevated,
        baseline_util_user=baseline_util,
    )

    stratum_label = (
        f"{location_label}, "
        f"{'symptomatic' if is_sympt else 'asymptomatic'}"
    )
    fig = make_single_strata_figure(
        cache,
        location=location,
        sex=sex,
        prior=is_sympt,
        highlight_age=age,
        stratum_label=stratum_label,
    )
    st.pyplot(fig, use_container_width=True)
