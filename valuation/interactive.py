"""Interactive scouting comparisons and explicit counterfactual scenarios."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from valuation.model import predict
from valuation.ui import player_card


def scenario_profiles(
    profile, years=3, goals_change=0, assists_change=0, minutes_change=0
):
    """Age and performance scenarios in a fixed market context, not time-series forecasts."""
    rows = []
    base = profile.iloc[0].to_dict()
    for year in range(years + 1):
        row = dict(base)
        row["age"] = float(base["age"]) + year
        if row["age"] > 45:
            break
        for feature, change in [
            ("goals", goals_change),
            ("assists", assists_change),
            ("minutes_played", minutes_change),
        ]:
            row[feature] = max(
                0, round(float(base[feature]) * (1 + change / 100) ** year)
            )
        row["scenario_year"] = year
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_resource
def load_forecast(path, modified):
    return joblib.load(path)


def render_lab(bundle, data, profile, model_name):
    compare_tab, scenario_tab, discover_tab, forecast_tab = st.tabs(
        [
            "Compare players",
            "What-if studio",
            "Scout & shortlist",
            "Next-season forecast",
        ]
    )
    latest = data[data.season == data.season.max()] if "season" in data else data
    with compare_tab:
        st.subheader("Build your comparison")
        if "player_id" not in latest:
            st.info("Select the real player database to compare named players.")
        else:
            catalog = (
                latest.sort_values("market_value_eur", ascending=False)
                .drop_duplicates("player_id")
                .set_index("player_id")
            )
            default = (
                catalog[catalog.player_club_id == 27].index.tolist()[:2]
                or catalog.index.tolist()[:2]
            )
            choices = st.multiselect(
                "Compare up to four players",
                catalog.index.tolist(),
                default=default,
                max_selections=4,
                format_func=lambda value: (
                    f"{catalog.loc[value, 'player_name']} · {catalog.loc[value, 'club_name']}"
                ),
            )
            if choices:
                players = catalog.loc[choices].reset_index()
                for column, (_, row) in zip(
                    st.columns(len(players)), players.iterrows()
                ):
                    column.markdown(player_card(row), unsafe_allow_html=True)
                results = predict(bundle, players, model_name)
                st.dataframe(
                    results[
                        [
                            "player_name",
                            "position",
                            "age",
                            "goals",
                            "assists",
                            "minutes_played",
                            "market_value_eur",
                            "estimated_value_eur",
                            "lower_90_eur",
                            "upper_90_eur",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                plot = results.melt(
                    id_vars="player_name",
                    value_vars=["market_value_eur", "estimated_value_eur"],
                    var_name="Value type",
                    value_name="Euros",
                )
                st.plotly_chart(
                    px.bar(
                        plot,
                        x="player_name",
                        y="Euros",
                        color="Value type",
                        barmode="group",
                        color_discrete_sequence=["#647c9e", "#fa4666"],
                    ),
                    width="stretch",
                )
                st.caption(
                    "Recorded values are historical labels. These players may have appeared in fitting; this comparison does not measure independent accuracy."
                )
                st.download_button(
                    "Export comparison",
                    results.to_csv(index=False),
                    "player_comparison.csv",
                    "text/csv",
                )
    with scenario_tab:
        st.subheader("Change the statistics")
        st.caption(
            "Counterfactual scenarios hold league and market-year context fixed. These are not validated forecasts of future transfer prices; injuries, contracts, market inflation and changes of club are not modeled."
        )
        if profile is None:
            st.info("Choose a player or estimate a custom profile first.")
        else:
            c1, c2, c3 = st.columns(3)
            growth_goals = c1.slider("Annual goal change (%)", -50, 100, 10, 5)
            growth_assists = c2.slider("Annual assist change (%)", -50, 100, 5, 5)
            growth_minutes = c3.slider("Annual playing-time change (%)", -50, 50, 0, 5)
            years = st.slider("Scenario horizon (years)", 1, 5, 3)
            scenarios = scenario_profiles(
                profile, years, growth_goals, growth_assists, growth_minutes
            )
            results = predict(bundle, scenarios, model_name)
            st.plotly_chart(
                px.line(
                    results,
                    x="scenario_year",
                    y=["estimated_value_eur", "lower_90_eur", "upper_90_eur"],
                    markers=True,
                    labels={
                        "scenario_year": "Scenario year",
                        "value": "Value (€)",
                        "variable": "Estimate / interval",
                    },
                    color_discrete_sequence=["#ff4768", "#647c9e", "#89a5cd"],
                ),
                width="stretch",
            )
            st.dataframe(
                results[
                    [
                        "scenario_year",
                        "age",
                        "goals",
                        "assists",
                        "minutes_played",
                        "estimated_value_eur",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            outside = [
                feature
                for feature, (low, high) in bundle["ranges"].items()
                if not results[feature].between(low, high).all()
            ]
            if outside:
                st.warning(
                    "Scenario exceeds the observed data range: " + ", ".join(outside)
                )
            st.download_button(
                "Export scenario",
                results.to_csv(index=False),
                "valuation_scenarios.csv",
                "text/csv",
            )
    with discover_tab:
        st.subheader("Filter players")
        if "player_id" not in latest:
            st.info("Select the real player database for scouting filters.")
        else:
            c1, c2, c3 = st.columns(3)
            age_range = c1.slider("Scouting age range", 15, 45, (18, 27))
            budget = c2.slider("Maximum recorded value (€m)", 1, 200, 50)
            positions = c3.multiselect(
                "Scouting positions",
                sorted(latest.position.unique()),
                default=["Forward"],
            )
            search = st.text_input(
                "Search name or club", placeholder="Player or club name…"
            )
            prospects = latest[
                latest.age.between(*age_range)
                & (latest.market_value_eur <= budget * 1e6)
                & latest.position.isin(positions)
            ]
            if search:
                prospects = prospects[
                    prospects.player_name.str.contains(search, case=False, regex=False)
                    | prospects.club_name.str.contains(
                        search, case=False, regex=False, na=False
                    )
                ]
            prospects = prospects.sort_values(
                "market_value_eur", ascending=False
            ).drop_duplicates("player_id")
            st.write(f"**{len(prospects)} players** match your filters")
            shortlist = st.multiselect(
                "Save a session shortlist",
                prospects.player_id.tolist(),
                format_func=lambda value: str(
                    prospects.set_index("player_id").loc[value, "player_name"]
                ),
            )
            view = (
                prospects[prospects.player_id.isin(shortlist)]
                if shortlist
                else prospects
            )
            st.dataframe(
                view[
                    [
                        "player_name",
                        "club_name",
                        "position",
                        "age",
                        "goals",
                        "assists",
                        "minutes_played",
                        "market_value_eur",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Export shortlist / results",
                view.to_csv(index=False),
                "scouting_shortlist.csv",
                "text/csv",
            )
    with forecast_tab:
        st.subheader("One season ahead")
        forecast_path = Path("artifacts/forecast_model.joblib")
        if not forecast_path.exists():
            st.info(
                "Build the forecast dataset and train the next-season artifact using the README commands."
            )
        elif profile is None:
            st.info("Select a player or enter a custom profile to forecast.")
        else:
            forecast = load_forecast(
                str(forecast_path), forecast_path.stat().st_mtime_ns
            )
            forecast_metrics = forecast["metrics"].loc[forecast["selected"]]
            st.caption(
                "This separate supervised model learns next-season recorded valuations from the preceding season's statistics. It is not the same-season valuation model."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Historical forecast test MAE",
                f"€{forecast_metrics.test_mae / 1e6:.2f}m",
            )
            c2.metric("Historical forecast test R²", f"{forecast_metrics.test_r2:.3f}")
            c3.metric("Forecast model", forecast["selected"])
            base = profile.copy()
            feature_season = int(
                base.iloc[0].get(
                    "season", bundle.get("context_defaults", {}).get("season", 2025)
                )
            )
            base["season"] = feature_season + 1
            result = predict(forecast, base)
            row = result.iloc[0]
            st.metric(
                f"Next-season estimate · {feature_season + 1}/{str(feature_season + 2)[-2:]}",
                f"€{row.estimated_value_eur / 1e6:.2f}m",
            )
            st.write(
                f"Historically calibrated 90% interval: €{row.lower_90_eur / 1e6:.2f}m – €{row.upper_90_eur / 1e6:.2f}m"
            )
            st.warning(
                "Forecasts beyond the last observed target season (2025/26) are unvalidated. The historical interval may not retain coverage. Consecutive-season player coverage introduces survivorship bias."
            )
            st.download_button(
                "Export next-season forecast",
                result.to_csv(index=False),
                "next_season_forecast.csv",
                "text/csv",
            )
