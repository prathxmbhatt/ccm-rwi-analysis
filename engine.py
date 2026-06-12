"""CCM-RWI engine + decision-zone heatmap for the Streamlit app."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parent / "reference_data"
final_male_df = pd.read_csv(_DATA / "final_male_df.csv")
final_female_df = pd.read_csv(_DATA / "final_female_df.csv")


DEFAULTS = {
    "util_well_nbs": 0.88,
    "util_well_bs": 0.83,
    "util_disabled": 0.64,
    "weight_disability": 0.36,
    "years_elevated": 5,
    "nh": {
        ("nonbrainstem", False): {"early": 0.003,   "late": 0.003, "p_death": 0.022, "p_disab": 0.06},
        ("brainstem",    False): {"early": 0.028,   "late": 0.028, "p_death": 0.023, "p_disab": 0.21},
        ("nonbrainstem", True):  {"early": 0.03985, "late": 0.003, "p_death": 0.022, "p_disab": 0.06},
        ("brainstem",    True):  {"early": 0.07099, "late": 0.028, "p_death": 0.023, "p_disab": 0.21},
    },
    "tx": {
        ("resection", "brainstem"):    {"p_death": 0.018, "p_disab": 0.29},
        ("resection", "nonbrainstem"): {"p_death": 0.010, "p_disab": 0.026},
    },
}


def _default_baseline_util(location):
    return DEFAULTS["util_well_bs"] if location == "brainstem" else DEFAULTS["util_well_nbs"]


def _life_tables(age, sex, baseline_util=1.0):
    df = final_male_df if sex == "male" else final_female_df
    yhl_col = "m_yhl" if sex == "male" else "f_yhl"
    life_col = "m_life_remain" if sex == "male" else "f_life_remain"
    life_remaining = df.loc[age, life_col]
    life_exp = age + life_remaining
    round_life_exp = math.ceil(life_exp) - 1
    df_life = df.loc[age:round_life_exp, yhl_col].copy().astype(float)
    df_life.loc[round_life_exp] = df_life.loc[round_life_exp] * (life_exp % 1)
    if baseline_util != 1.0:
        df_life = df_life * baseline_util
    return df_life, life_remaining, round_life_exp


def _forward_markov(age, sex, initial_state, init_tr, init_impact,
                    rate_fn, p_death, p_disability,
                    weight_disability, baseline_util, years_elevated=5):
    df_life, life_remaining, round_life_exp = _life_tables(age, sex, baseline_util)
    sums = df_life.values.astype(float)
    n_years = len(sums)

    remaining = np.zeros(n_years + 1)
    acc = 0.0
    for i in range(n_years - 1, -1, -1):
        acc += sums[i]
        remaining[i] = acc

    life_exp = age + life_remaining
    frac_last_year = life_exp - round_life_exp
    if frac_last_year <= 0.0 or frac_last_year > 1.0:
        frac_last_year = 1.0

    p_poor = p_death + p_disability
    if p_poor > 0:
        sub_impact = (1.0 * p_death + weight_disability * p_disability) / p_poor
    else:
        sub_impact = 0.0

    # State 0 = baseline; states 1..years_elevated = elevated phase
    n_states = years_elevated + 1
    p = np.zeros(n_states)
    p[min(initial_state, years_elevated)] = 1.0 - init_tr

    rwi = init_tr * init_impact * sums.sum()
    cum_event = init_tr

    for year in range(1, n_years + 1):
        new_p = np.zeros(n_states)
        terminal_this_year = 0.0
        year_fraction = frac_last_year if year == n_years else 1.0

        for s in range(n_states):
            if p[s] == 0:
                continue
            rate_full = rate_fn(year, s)
            rate_full = max(0.0, min(rate_full, 0.999))
            if year_fraction == 1.0:
                rate = rate_full
            else:
                rate = 1.0 - (1.0 - rate_full) ** year_fraction

            p_no_bleed = 1.0 - rate
            p_recovery = rate * (1.0 - p_poor)
            p_terminal = rate * p_poor

            terminal_this_year += p[s] * p_terminal

            if s == 0:
                new_p[0] += p[s] * p_no_bleed
            elif s < years_elevated:
                new_p[s + 1] += p[s] * p_no_bleed
            else:  # s == years_elevated → reverts to baseline
                new_p[0] += p[s] * p_no_bleed
            new_p[1] += p[s] * p_recovery

        # Start-of-year hazard convention (matches paper_release/ccm_rwi.py):
        # a poor outcome in loop-year `year` forfeits remaining utility-life
        # from that year onward (remaining[year - 1]), giving the observation
        # arm a year-0 term symmetric with the treatment arm's upfront event.
        rwi += terminal_this_year * sub_impact * remaining[year - 1]
        cum_event += terminal_this_year
        p = new_p

    return rwi, cum_event


def compute_patient_risks(
    age, sex, location, presentation_state,
    nh_baseline_rate, nh_elevated_rate,
    p_death_given_bleed, p_disability_given_bleed,
    tx_p_death, tx_p_disability,
    weight_disability, baseline_util, years_elevated,
):
    # If patient is further out from bleed than the elevated window, they are back at baseline.
    if presentation_state == 0 or presentation_state > years_elevated:
        initial_state = 0
    else:
        initial_state = presentation_state

    def obs_rate(year, state):
        return nh_baseline_rate if state == 0 else nh_elevated_rate

    obs_rwi, obs_cum = _forward_markov(
        age, sex, initial_state, 0.0, 1.0, obs_rate,
        p_death_given_bleed, p_disability_given_bleed,
        weight_disability, baseline_util, years_elevated,
    )

    p_poor_tx = tx_p_death + tx_p_disability
    if p_poor_tx > 0:
        tx_init_impact = (1.0 * tx_p_death + weight_disability * tx_p_disability) / p_poor_tx
    else:
        tx_init_impact = 0.0

    def res_rate(year, state):
        return 0.0

    res_rwi, res_cum = _forward_markov(
        age, sex, initial_state, p_poor_tx, tx_init_impact, res_rate,
        p_death_given_bleed, p_disability_given_bleed,
        weight_disability, baseline_util, years_elevated,
    )

    return obs_rwi, obs_cum, res_rwi, res_cum


def _compute_cell(age, sex, initial_state,
                  nh_baseline_rate, nh_elevated_rate,
                  p_death, p_disab,
                  tx_p_death, tx_p_disab,
                  weight_disability, baseline_util, years_elevated):
    def obs_rate(year, state):
        return nh_baseline_rate if state == 0 else nh_elevated_rate
    obs_rwi, obs_cum = _forward_markov(
        age, sex, initial_state, 0.0, 1.0, obs_rate,
        p_death, p_disab, weight_disability, baseline_util, years_elevated,
    )
    p_poor_tx = tx_p_death + tx_p_disab
    if p_poor_tx > 0:
        tx_init_impact = (tx_p_death + weight_disability * tx_p_disab) / p_poor_tx
    else:
        tx_init_impact = 0.0
    def res_rate(year, state):
        return 0.0
    res_rwi, res_cum = _forward_markov(
        age, sex, initial_state, p_poor_tx, tx_init_impact, res_rate,
        p_death, p_disab, weight_disability, baseline_util, years_elevated,
    )
    return obs_rwi, obs_cum, res_rwi, res_cum


def build_decision_cache(user_location=None, user_overrides=None,
                         weight_disability=None, years_elevated=5,
                         baseline_util_user=None):
    """Build decision cache. If user_location and user_overrides are provided,
    use the user's modified parameters for that location; other location uses
    literature defaults. weight_disability and years_elevated apply globally.
    """
    if weight_disability is None:
        weight_disability = DEFAULTS["weight_disability"]
    if user_overrides is None:
        user_overrides = {}

    cache = {}
    for sex in ("male", "female"):
        for loc in ("brainstem", "nonbrainstem"):
            if loc == user_location:
                nh_baseline_rate = user_overrides.get("nh_baseline_rate",
                                                      DEFAULTS["nh"][(loc, False)]["late"])
                nh_elevated_rate = user_overrides.get("nh_elevated_rate",
                                                      DEFAULTS["nh"][(loc, True)]["early"])
                p_death = user_overrides.get("p_death_given_bleed",
                                              DEFAULTS["nh"][(loc, False)]["p_death"])
                p_disab = user_overrides.get("p_disability_given_bleed",
                                              DEFAULTS["nh"][(loc, False)]["p_disab"])
                tx_p_death = user_overrides.get("tx_p_death",
                                                 DEFAULTS["tx"][("resection", loc)]["p_death"])
                tx_p_disab = user_overrides.get("tx_p_disab",
                                                 DEFAULTS["tx"][("resection", loc)]["p_disab"])
                baseline_util = (baseline_util_user if baseline_util_user is not None
                                 else _default_baseline_util(loc))
            else:
                nh_baseline_rate = DEFAULTS["nh"][(loc, False)]["late"]
                nh_elevated_rate = DEFAULTS["nh"][(loc, True)]["early"]
                p_death = DEFAULTS["nh"][(loc, False)]["p_death"]
                p_disab = DEFAULTS["nh"][(loc, False)]["p_disab"]
                tx_p_death = DEFAULTS["tx"][("resection", loc)]["p_death"]
                tx_p_disab = DEFAULTS["tx"][("resection", loc)]["p_disab"]
                baseline_util = _default_baseline_util(loc)

            for prior in (False, True):
                initial_state = 1 if prior else 0
                for age in range(15, 86):
                    obs_rwi, obs_cum, res_rwi, res_cum = _compute_cell(
                        age, sex, initial_state,
                        nh_baseline_rate, nh_elevated_rate,
                        p_death, p_disab,
                        tx_p_death, tx_p_disab,
                        weight_disability, baseline_util, years_elevated,
                    )
                    cache[(age, sex, loc, prior)] = (res_cum < obs_cum, res_rwi < obs_rwi)
    return cache


CLR_BG       = "#F7F9FC"
CLR_PANEL    = "#FFFFFF"
CLR_TEXT     = "#1A202C"
CLR_SUB      = "#4A5568"
CLR_GRID     = "#DDE3EC"
CLR_DIV      = "#CBD5E0"
CLR_OBS      = "#2563EB"
CLR_RES      = "#DC2626"
CLR_GREEN    = "#2D7D46"
CLR_GREEN_L  = "#EAF4EE"
CLR_YELLOW   = "#B45309"
CLR_YELLOW_L = "#FEF9EC"
CLR_TEAL     = "#0E7490"
CLR_TEAL_L   = "#ECFEFF"
CLR_BLUE_DK  = "#1E40AF"
CLR_BLUE_L   = "#EFF6FF"


def make_single_strata_figure(cache, location, sex, prior, highlight_age=None,
                              stratum_label=""):
    """Single-row horizontal strip showing decision zones across age for ONE
    specific stratum + sex. Clean layout, no overlapping legend/labels."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": CLR_TEXT,
    })

    ages = list(range(20, 81))
    zone_colors = {0: CLR_BLUE_L, 1: CLR_YELLOW_L, 2: CLR_GREEN_L, -1: CLR_TEAL_L}

    # Build the decision row
    row = np.zeros(len(ages), dtype=np.int8)
    for j, a in enumerate(ages):
        c, r = cache[(a, sex, location, prior)]
        if c and r: row[j] = 2
        elif not c and not r: row[j] = 0
        elif c and not r: row[j] = 1
        else: row[j] = -1

    def _crossover_idx(treat_mask):
        for j in range(1, len(treat_mask)):
            if treat_mask[j-1] and not treat_mask[j]:
                return j
        return None

    cum_treat = np.isin(row, [2, 1])
    rwi_treat = np.isin(row, [2, -1])
    j_cum = _crossover_idx(cum_treat)
    j_rwi = _crossover_idx(rwi_treat)

    fig = plt.figure(figsize=(12, 4.0), facecolor=CLR_BG)
    gs = fig.add_gridspec(1, 1, left=0.04, right=0.97, top=0.72, bottom=0.32)
    ax = gs.subplots()
    ax.set_facecolor(CLR_PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Color the strip
    for j in range(len(ages)):
        ax.add_patch(plt.Rectangle(
            (j - 0.5, -0.5), 1, 1,
            facecolor=zone_colors[row[j]],
            edgecolor="none", linewidth=0, zorder=1,
        ))

    # Crossover annotations
    if j_cum is not None:
        x = j_cum - 0.5
        ax.plot([x, x], [-0.55, 0.55], color=CLR_TEXT,
                linewidth=1.6, linestyle="-", zorder=4)
        ax.text(x, -0.62, f"Cum crossover: {ages[j_cum]}",
                ha="center", va="bottom", fontsize=9,
                color=CLR_TEXT, fontweight="700", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=CLR_PANEL, edgecolor=CLR_DIV, linewidth=0.6))
    if j_rwi is not None:
        x = j_rwi - 0.5
        ax.plot([x, x], [-0.55, 0.55], color=CLR_TEXT,
                linewidth=1.6, linestyle=(0, (3, 2)), zorder=4)
        ax.text(x, 0.62, f"RWI crossover: {ages[j_rwi]}",
                ha="center", va="top", fontsize=9,
                color=CLR_TEXT, fontweight="700", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=CLR_PANEL, edgecolor=CLR_DIV, linewidth=0.6))

    # Patient's age highlight
    if highlight_age is not None and 20 <= highlight_age <= 80:
        j_age = highlight_age - 20
        ax.add_patch(plt.Rectangle(
            (j_age - 0.5, -0.5), 1, 1,
            facecolor="none", edgecolor=CLR_RES,
            linewidth=3, zorder=6,
        ))

    ax.set_xlim(-0.5, len(ages) - 0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks(range(0, len(ages), 5))
    ax.set_xticklabels([str(ages[k]) for k in range(0, len(ages), 5)],
                       fontsize=9.5, color=CLR_SUB)
    ax.set_yticks([])
    ax.set_xlabel("Age at decision (years)", fontsize=10,
                  color=CLR_SUB, labelpad=10)

    fig.text(0.5, 0.92,
             f"Decision-zone recommendations across age — {stratum_label}, {sex}",
             ha="center", va="top",
             fontsize=12, fontweight="700", color=CLR_TEXT)

    # Legend below the chart, well-separated from x-axis label
    legend_handles = []
    used_zones = set(np.unique(row).tolist())
    zone_legend = [
        (0, CLR_BLUE_L, CLR_BLUE_DK, "Both policies observe"),
        (1, CLR_YELLOW_L, CLR_YELLOW, "Cum treat / RWI observe"),
        (2, CLR_GREEN_L, CLR_GREEN, "Both policies treat"),
        (-1, CLR_TEAL_L, CLR_TEAL, "Cum observe / RWI treat"),
    ]
    for zone, bg, fg, lbl in zone_legend:
        if zone in used_zones:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=bg,
                                                edgecolor=fg, linewidth=0.8, label=lbl))
    legend_handles.append(plt.Line2D([0], [0], color=CLR_TEXT, linewidth=1.6,
                                     linestyle="-", label="Cum crossover"))
    legend_handles.append(plt.Line2D([0], [0], color=CLR_TEXT, linewidth=1.6,
                                     linestyle=(0, (3, 2)), label="RWI crossover"))
    if highlight_age is not None:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                            edgecolor=CLR_RES, linewidth=3,
                                            label="Patient age"))
    leg = fig.legend(handles=legend_handles,
                     loc="lower center", ncol=len(legend_handles),
                     fontsize=9, frameon=False,
                     bbox_to_anchor=(0.5, 0.02))
    for text in leg.get_texts():
        text.set_color(CLR_TEXT)

    return fig
