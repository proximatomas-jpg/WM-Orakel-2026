import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

# ==========================================
# 1. SEITEN-KONFIGURATION (Muss ganz oben stehen)
# ==========================================
st.set_page_config(page_title="WM 2026 Orakel", page_icon="⚽", layout="centered")

# ==========================================
# 2. DATEN LADEN (Mit Cache!)
# ==========================================
# @st.cache_data sorgt dafür, dass die riesige CSV nicht bei jedem Klick neu geladen wird!
@st.cache_data
def load_and_prep_data():
    # TRAGE HIER DEINEN PFAD EIN:
    path = Path(r"results.csv")
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= "2023-01-01"].copy()
    
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    
    # Recency-Gewichtung (Neuere Spiele zählen mehr)
    latest_date = df["date"].max()
    age_days = (latest_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = 0.5 ** (age_days / 365)
    
    return df

df = load_and_prep_data()
all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))

global_mean_goals = np.average(
    pd.concat([df["home_score"], df["away_score"]]),
    weights=pd.concat([df["weight"], df["weight"]])
)

# ==========================================
# 3. BERECHNUNGS-LOGIK (Vereinfacht für Web)
# ==========================================
def get_team_stats(team):
    games = df[(df["home_team"] == team) | (df["away_team"] == team)]
    if games.empty:
        return 1.0, 1.0, 0
    
    scored = np.where(games["home_team"] == team, games["home_score"], games["away_score"])
    conceded = np.where(games["home_team"] == team, games["away_score"], games["home_score"])
    weights = games["weight"].to_numpy()
    
    avg_scored = np.average(scored, weights=weights) if weights.sum() > 0 else global_mean_goals
    avg_conceded = np.average(conceded, weights=weights) if weights.sum() > 0 else global_mean_goals
    
    attack = avg_scored / global_mean_goals
    defense = avg_conceded / global_mean_goals
    return attack, max(defense, 0.2), weights.sum()

# ==========================================
# 4. DAS WEB-INTERFACE
# ==========================================
st.title("🏆 WM 2026 Orakel")
st.markdown("Statistisches Poisson-Modell basierend auf historischen Länderspielen.")

# Layout in zwei Spalten für die Teamauswahl (Sieht auf dem Handy super aus)
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Team A (Heim)", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 0)

with col2:
    away_team = st.selectbox("Team B (Auswärts)", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

# Optionen
st.markdown("### ⚙️ Einstellungen")
has_home_advantage = st.checkbox("Team A hat echten Heimvorteil", value=False)
is_ko_phase = st.checkbox("K.O.-Phase (Entscheidung erzwingen)", value=False)

# Berechnen-Button
if st.button("⚽ Spiel simulieren", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Ein Team kann nicht gegen sich selbst spielen!")
    else:
        with st.spinner("Berechne Millionen von Möglichkeiten..."):
            # Stärken abrufen
            att_home, def_home, games_home = get_team_stats(home_team)
            att_away, def_away, games_away = get_team_stats(away_team)
            
            # Heimvorteil einrechnen oder neutraler Boden
            if has_home_advantage:
                lambda_home = att_home * def_away * global_mean_goals * 1.15 # 15% Heimbonus
                lambda_away = att_away * def_home * global_mean_goals * 0.85 # 15% Auswärtsmalus
            else:
                lambda_home = att_home * def_away * global_mean_goals
                lambda_away = att_away * def_home * global_mean_goals
                
            # Poisson Matrix (bis 6 Tore)
            max_g = 6
            matrix = np.outer(poisson.pmf(np.arange(max_g+1), lambda_home), poisson.pmf(np.arange(max_g+1), lambda_away))
            
            # Ergebnisse extrahieren
            p_home, p_away = np.unravel_index(np.argmax(matrix), matrix.shape)
            highest_prob = matrix[p_home, p_away] * 100
            
            win_home = np.sum(np.tril(matrix, -1)) * 100
            draw = np.sum(np.diag(matrix)) * 100
            win_away = np.sum(np.triu(matrix, 1)) * 100
            
            # --- ERGEBNISSE ANZEIGEN ---
            st.divider()
            
            # Große Anzeige des wahrscheinlichsten Ergebnisses
            st.markdown(f"<h2 style='text-align: center;'>Tipp: {p_home} : {p_away}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gray;'>Wahrscheinlichkeit: {highest_prob:.1f}%</p>", unsafe_allow_html=True)
            
            # Metriken in 3 Spalten (Sieg, Unentschieden, Sieg)
            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"Sieg {home_team}", value=f"{win_home:.1f}%")
            m2.metric(label="Unentschieden", value=f"{draw:.1f}%")
            m3.metric(label=f"Sieg {away_team}", value=f"{win_away:.1f}%")
            
            st.info(f"**Erwartete Tore (λ):** {home_team} **{lambda_home:.2f}** | {away_team} **{lambda_away:.2f}**")
            
            # K.o.-Phase Logik
            if is_ko_phase:
                st.divider()
                st.markdown("### 🏆 Wer kommt weiter? (inkl. Verlängerung)")
                
                # Sehr simple Stärkerechnung für den Tie-Breaker
                home_strength = att_home / def_home
                away_strength = att_away / def_away
                home_ko_weight = home_strength / (home_strength + away_strength)
                
                adv_home = win_home + (draw * home_ko_weight)
                adv_away = win_away + (draw * (1 - home_ko_weight))
                
                if adv_home > adv_away:
                    st.success(f"**{home_team}** zieht ins nächste Spiel ein! (Chance: {adv_home:.1f}%)")
                else:
                    st.success(f"**{away_team}** zieht ins nächste Spiel ein! (Chance: {adv_away:.1f}%)")
