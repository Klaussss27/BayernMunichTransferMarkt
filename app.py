import pandas as pd
from pathlib import Path
import hashlib
import json
import joblib
from html import escape
import plotly.express as px
import streamlit as st

from valuation.data import POSITIONS, demo_data, validate_data, load_database
from valuation.model import predict, train
from valuation.ui import STYLES, portrait, player_card
from valuation.interactive import render_lab

st.set_page_config(page_title="Bayern · Player Valuation", page_icon="⚽", layout="wide")
st.markdown(STYLES, unsafe_allow_html=True)
st.markdown('<div class="topbar"><div class="wordmark">SCOUT<span> / </span>BAYERN</div><div class="status">● &nbsp; PLAYER INTELLIGENCE</div></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">THE GAME, THROUGH A DIFFERENT LENS</div>', unsafe_allow_html=True)
st.markdown('<h1>See the player.<br><span style="color:#91a1b7">Understand the value.</span></h1>', unsafe_allow_html=True)
st.markdown('<div class="intro">From the pitch to the numbers. Explore player profiles, discover your squad, and put a statistical perspective on market value.</div>', unsafe_allow_html=True)

database_path = Path("data/bundesliga.sqlite")
model_path = Path("artifacts/bundesliga_model.joblib")
provenance_path = Path("data/provenance.json")
real_label = "Trained Bundesliga database"
if Path("data/european_players.sqlite").exists() and Path("artifacts/european_model.joblib").exists():
    database_path = Path("data/european_players.sqlite")
    model_path = Path("artifacts/european_model.joblib")
    provenance_path = Path("data/european_provenance.json")
    real_label = "Trained European database"
real_available = database_path.exists() and model_path.exists()
with st.sidebar:
    st.header("SCOUT / BAYERN")
    st.caption("FOOTBALL INTELLIGENCE STUDIO")
    options = ([real_label] if real_available else []) + ["Synthetic demonstration", "Upload CSV"]
    source = st.radio("Training data", options)
    uploaded = st.file_uploader("Training CSV", type="csv") if source == "Upload CSV" else None
    st.divider()
    st.write("**Features**")
    st.caption("Age · Goals · Assists · Position · Minutes played")
    st.write("**Models**")
    st.caption("Random Forest · XGBoost · Median baseline")
    st.caption("No affiliation with FC Bayern Munich or the Bundesliga.")
    st.divider()
    st.caption("01 / PLAYER VALUATION\n\n02 / SQUAD ROOM\n\n03 / MODEL INSIGHTS\n\n04 / DATA EXPLORER")

if source == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV to begin. Download the example below for the expected format.")
    st.download_button("Download example CSV", demo_data(100).to_csv(index=False), "synthetic_players.csv", "text/csv")
    st.stop()
try:
    data = load_database(database_path) if source == real_label else validate_data(pd.read_csv(uploaded)) if uploaded else validate_data(demo_data())
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()

is_demo = source == "Synthetic demonstration"
if is_demo:
    st.warning("DEMO DATA · All players and market values are fictional. Results demonstrate the software, not real-world accuracy.")
elif source == real_label:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    st.info(f"REAL DATA · {provenance['competition']} · {provenance['season']} · {len(data):,} observations · Valuations {provenance['valuation_date_min']} to {provenance['valuation_date_max']}. Historical snapshot; not live prices.")
else:
    st.info("CSV values must be in euros, with one independent observation per player. Accuracy depends on your data's quality and coverage.")


@st.cache_resource(show_spinner="Training models and evaluating held-out players…")
def cached_train(frame):
    return train(frame)


@st.cache_resource(show_spinner="Loading trained Bundesliga model…")
def cached_load(path, modified):
    return joblib.load(path)


if source == real_label:
    bundle = cached_load(str(model_path), model_path.stat().st_mtime_ns)
    fingerprint = hashlib.sha256(data.to_csv(index=False).encode("utf-8")).hexdigest()
    if fingerprint != bundle["metadata"]["dataset_sha256"]:
        st.error(f"The database has changed since model training. Run train.py --database {database_path} --output {model_path} to update it.")
        st.stop()
else:
    bundle = cached_train(data)
with st.sidebar:
    st.divider()
    model_choices = bundle["metrics"].sort_values("validation_mae").index.tolist()
    active_model = st.selectbox("Valuation model", model_choices, index=model_choices.index(bundle["selected"]))
    st.caption(f"Validation-selected recommendation: {bundle['selected']}")
active_predictor = active_model if active_model in bundle.get("models", {}) else None
metrics = bundle["metrics"].loc[active_model]
if st.session_state.get("data_source") != source:
    st.session_state["profile"] = None
    st.session_state["scouted_id"] = None
    st.session_state["data_source"] = source

latest = data[data.season == data.season.max()].copy() if "season" in data else data.copy()
selected_player = None
if source == real_label and "player_id" in latest:
    scouting_clubs = sorted(latest.club_name.dropna().unique())
    scouting_club = st.selectbox("Scouting club", scouting_clubs, index=scouting_clubs.index("Bayern Munich") if "Bayern Munich" in scouting_clubs else 0)
    roster = latest[latest.club_name == scouting_club].sort_values("market_value_eur", ascending=False)
    if roster.empty:
        roster = latest.nlargest(30, "market_value_eur")
    roster = roster.drop_duplicates("player_id")
    lookup = roster.set_index("player_id")
    chosen = st.selectbox("SCOUT A PLAYER · 2025/26 SNAPSHOT", roster.player_id.tolist(), format_func=lambda value: str(lookup.loc[value, "player_name"]))
    selected_player = lookup.loc[chosen]
    featured_profile = pd.DataFrame([selected_player.to_dict()])
    featured_profile["player_id"] = chosen
    if st.session_state.get("scouted_id") != chosen:
        st.session_state["profile"] = featured_profile
        st.session_state["scouted_id"] = chosen
    name_html = escape(str(selected_player.player_name))
    club_html = escape(str(selected_player.club_name))
    st.markdown(f'''<div class="hero"><div><div class="hero-tag">FEATURED PLAYER / {club_html.upper()}</div>
<div class="hero-name">{name_html}</div><div class="hero-sub">{club_html} &nbsp; / &nbsp; {escape(str(selected_player.position))} &nbsp; / &nbsp; {int(selected_player.age)} years</div>
<div class="hero-stat">€{selected_player.market_value_eur / 1e6:.1f}m <span>RECORDED MARKET VALUE · 2025/26</span></div>
<div class="hero-sub" style="margin-top:14px">{int(selected_player.goals)} goals &nbsp; · &nbsp; {int(selected_player.assists)} assists &nbsp; · &nbsp; {int(selected_player.minutes_played):,} minutes</div></div>{portrait(chosen, selected_player.player_name, 190)}</div>''', unsafe_allow_html=True)
    st.caption("Historical season statistics and recorded valuation. Portraits are from the source player profiles and may show a different kit or age.")
a, b, c, d = st.columns(4)
a.metric("Dataset observations", f"{len(data):,}")
b.metric("Active model", active_model.replace(" depth 5", " d5"))
c.metric("Test MAE", f"€{metrics.test_mae / 1e6:.2f}m")
d.metric("Test R²", f"{metrics.test_r2:.3f}")

valuation_tab, squad_tab, models_tab, data_tab, lab_tab = st.tabs(["01  Player valuation", "02  Squad room", "03  Model insights", "04  Data explorer", "05  Interactive lab"])
with valuation_tab:
    if bundle.get("context_defaults"):
        st.caption("Profiles default to Bundesliga, 2025/26 season context. Batch CSVs may provide season (starting year) and competition_id (L1, GB1, ES1, IT1, FR1). The model has no 2026/27 training data.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Build a player profile")
        live_estimates = st.toggle("Live valuation — update as I change inputs", value=True)
        with st.container(border=True):
            defaults = selected_player if selected_player is not None else {"player_name": "", "position": "Forward", "age": 24, "minutes_played": 2200, "goals": 15, "assists": 8}
            form_suffix = str(chosen) if selected_player is not None else source
            name = st.text_input("Player name (optional)", value=str(defaults["player_name"]), placeholder="Enter a player name", key=f"name_{form_suffix}")
            position = st.selectbox("Playing position", POSITIONS, index=POSITIONS.index(defaults["position"]), key=f"position_{form_suffix}")
            age = st.slider("Age", 15.0, 45.0, float(defaults["age"]), .1, key=f"age_{form_suffix}")
            minutes = st.number_input("Minutes played in the season", 0, 10000, int(defaults["minutes_played"]), 90, key=f"minutes_{form_suffix}")
            col1, col2 = st.columns(2)
            goals = col1.number_input("Goals", 0, 150, int(defaults["goals"]), key=f"goals_{form_suffix}")
            assists = col2.number_input("Assists", 0, 150, int(defaults["assists"]), key=f"assists_{form_suffix}")
            submitted = st.button("Estimate market value", type="primary", width="stretch")
        if submitted or live_estimates:
            st.session_state["profile"] = pd.DataFrame([{"player_name": name or "Player profile", "position": position, "age": age, "minutes_played": minutes, "goals": goals, "assists": assists}])
    with right:
        st.subheader("Valuation report")
        profile = st.session_state.get("profile")
        if profile is None:
            st.info("Set a player's season statistics and select Estimate market value.")
        else:
            result = predict(bundle, profile, active_predictor)
            row = result.iloc[0]
            st.write(f"**{row.player_name}** · {row.position} · {int(row.age)} years")
            st.metric("Estimated market value" + (" · DEMO" if is_demo else ""), f"€{row.estimated_value_eur / 1e6:,.2f}m")
            st.write(f"90% calibrated interval: **€{row.lower_90_eur / 1e6:,.2f}m – €{row.upper_90_eur / 1e6:,.2f}m**")
            st.caption("The interval targets marginal coverage for players drawn from the same distribution. It is not a guarantee for this individual player.")
            outside = [feature for feature, (low, high) in bundle["ranges"].items() if not low <= row[feature] <= high]
            if outside:
                st.warning("Outside the training data range: " + ", ".join(outside) + ". Predictions may be unreliable.")
            st.download_button("Download valuation CSV", result.to_csv(index=False), "player_valuation.csv", "text/csv")
        st.divider()
        st.caption("Values reflect statistical associations. Contract length, injuries, international reputation, wages, and transfer negotiations are not modeled.")
    st.subheader("Value a squad")
    batch = st.file_uploader("Upload player profiles (same feature columns, no market value required)", type="csv", key="batch")
    if batch:
        try:
            output = predict(bundle, pd.read_csv(batch), active_predictor)
            st.dataframe(output, hide_index=True, width="stretch")
            st.download_button("Download squad valuations", output.to_csv(index=False), "squad_valuations.csv", "text/csv")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(str(exc))

with squad_tab:
    st.subheader("The squad room")
    st.write("Familiar faces. A closer look at the numbers.")
    if "player_id" not in latest or "club_name" not in latest:
        st.info("Select the trained database to explore real players and their portraits.")
    else:
        clubs = sorted(latest.club_name.dropna().unique())
        club = st.selectbox("Choose a club", clubs, index=clubs.index("Bayern Munich") if "Bayern Munich" in clubs else 0)
        squad = latest[latest.club_name == club].sort_values("market_value_eur", ascending=False).drop_duplicates("player_id")
        st.caption("Recorded source values from 2025/26, not model estimates or current transfer prices. Players may have appeared in training; this is not an independent squad evaluation.")
        full_squad = st.toggle("Show full squad", value=False)
        shown = squad if full_squad else squad.head(12)
        for offset in range(0, len(shown), 4):
            for column, (_, player) in zip(st.columns(4), shown.iloc[offset:offset + 4].iterrows()):
                column.markdown(player_card(player), unsafe_allow_html=True)
        st.caption("Player portraits: Transfermarkt image URLs supplied by the dataset. Missing photographs display initials.")
        st.download_button("Download squad statistics", squad.to_csv(index=False), "squad_statistics.csv", "text/csv")
        if st.toggle("Estimate this entire squad", value=False):
            squad_results = predict(bundle, squad, active_predictor)
            st.dataframe(squad_results[["player_name", "market_value_eur", "estimated_value_eur", "lower_90_eur", "upper_90_eur"]], hide_index=True, width="stretch")
            st.download_button("Export squad estimates", squad_results.to_csv(index=False), "squad_estimates.csv", "text/csv")

with models_tab:
    st.subheader("Measured on unseen players")
    st.write("Selection uses validation MAE. The final test set is reserved for reporting; calibration players determine the prediction interval.")
    st.caption(bundle["metadata"].get("evaluation", "Independent player observations"))
    if bundle.get("bundesliga_test"):
        league = bundle["bundesliga_test"]
        st.write(f"Bundesliga test subset: **{league['rows']} observations** · MAE **€{league['mae'] / 1e6:.2f}m** · R² **{league['r2']:.3f}**")
    display = bundle["metrics"].rename(columns={"validation_mae": "Validation MAE (€)", "test_mae": "Test MAE (€)", "test_rmse": "Test RMSE (€)", "test_r2": "Test R²"})
    st.dataframe(display.style.format(precision=2), width="stretch")
    st.plotly_chart(px.bar(bundle["metrics"].reset_index().sort_values("validation_mae"), x="validation_mae", y="model", orientation="h", title="Model leaderboard · validation MAE", labels={"validation_mae": "Validation MAE (€)", "model": "Model"}, color_discrete_sequence=["#ff4768"]), width="stretch")
    if bundle["metadata"].get("test_caveat"):
        st.caption(bundle["metadata"]["test_caveat"])
    lcol, rcol = st.columns(2)
    chart = px.scatter(bundle["test_predictions"], x="actual", y="predicted", title="Actual vs estimated market value", labels={"actual": "Actual (€)", "predicted": "Predicted (€)"})
    limit = float(bundle["test_predictions"].max().max())
    chart.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit, line=dict(dash="dash", color="#dc143c"))
    lcol.plotly_chart(chart, width="stretch")
    rcol.plotly_chart(px.bar(bundle["importance"], x="mae_increase", y="feature", orientation="h", title="Permutation feature importance", labels={"mae_increase": "Increase in test MAE (€)", "feature": "Feature"}), width="stretch")
    st.caption("Importance measures predictive association, not causation; negative values can occur from sampling variation.")
    if bundle.get("context_defaults"):
        st.caption("Calibration uses the preceding season. Distribution changes between seasons can reduce interval coverage; the observed test coverage below measures this limitation.")
    st.write(f"Observed test interval coverage: **{bundle['coverage']:.1%}** · Target: **90%**")
    st.json(bundle["metadata"])

with data_tab:
    st.subheader("Training data")
    positions = st.multiselect("Filter positions", POSITIONS, default=POSITIONS)
    filtered = data[data.position.isin(positions)]
    if "season" in filtered:
        seasons = st.multiselect("Filter seasons (starting year)", sorted(data.season.unique()), default=sorted(data.season.unique())[-1:])
        leagues = st.multiselect("Filter leagues", sorted(data.competition_id.unique()), default=sorted(data.competition_id.unique()))
        filtered = filtered[filtered.season.isin(seasons) & filtered.competition_id.isin(leagues)]
    st.dataframe(filtered, hide_index=True, width="stretch")
    st.plotly_chart(px.scatter(filtered, x="age", y="market_value_eur", color="position", size="minutes_played", hover_name="player_name" if "player_name" in data else None, title="Age, playing time, and market value"), width="stretch")
    st.download_button("Download filtered dataset", filtered.to_csv(index=False), "synthetic_players.csv" if is_demo else "training_players.csv", "text/csv")

with lab_tab:
    render_lab(bundle, data, st.session_state.get("profile"), active_predictor)
    if "player_id" in data and selected_player is not None:
        st.subheader("Recorded valuation history")
        history = data[data.player_id == chosen].sort_values("season")
        st.plotly_chart(px.line(history, x="season", y="market_value_eur", markers=True, title=str(selected_player.player_name), labels={"season": "Season starting year", "market_value_eur": "Recorded valuation (€)"}), width="stretch")
        st.caption("Historical observations from the source dataset, not model predictions.")
