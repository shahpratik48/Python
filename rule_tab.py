from dash import dcc, html, Input, Output, State, no_update

from styles import metadata_dashboard_styles as mds
from utils.helpers import build_rule_formatted_sections
from utils.data import get_narrative_columns_for_insight, get_sample_narrative_for_insight


def layout(df_rules):
    distinct_insights = (
        df_rules.get("insight_type", df_rules.iloc[:, 0])
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if df_rules is not None and not df_rules.empty
        else []
    )
    distinct_insights = sorted(distinct_insights)

    return dcc.Tab(
        label="Insight Agent Search",
        value="tab-rule",
        children=html.Div(
            [
                html.H3("Insight Agent Search", style=mds["section_header_style"]),
                # ── Search toolbar ───────────────────────────────────────────
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Insight Rule", style=mds["dropdown_label_style"]),
                                dcc.Dropdown(
                                    id="rule-search",
                                    options=[{"label": c, "value": c} for c in distinct_insights],
                                    placeholder="Search Insight Rule...",
                                    clearable=True,
                                    style={**mds["dropdown_style"], "width": "480px"},
                                ),
                            ],
                        ),
                    ],
                    style={
                        **mds["toolbar_style"],
                        "justifyContent": "center",
                    },
                ),
                # ── Results ──────────────────────────────────────────────────
                html.Div(
                    id="rule-results",
                    children=[],
                    style={
                        **mds["content_card_style"],
                        "width": "100%",
                        "maxWidth": "1100px",
                        "margin": "0 auto",
                    },
                ),
            ],
            style=mds["table_container_style"],
        ),
    )


def register_callbacks(app, df_rules):
    @app.callback(
        Output("rule-results", "children"),
        Input("rule-search", "value"),
        State("rule-search", "search_value"),
    )
    def update_rule_results(selected_value, search_value):
        if df_rules is None or df_rules.empty:
            return html.P("No rule data available.")

        data = df_rules
        if selected_value:
            mask = data.get("insight_type", data.iloc[:, 0]).astype(str) == str(selected_value)
            filtered = data.loc[mask]
        else:
            sv = (search_value or "").strip()
            if not sv:
                return []
            insight_series = data.get("insight_type", data.iloc[:, 0]).astype(str)
            filtered = data[insight_series.str.contains(sv, case=False, na=False)]

        if filtered.empty:
            return html.P("No matching insights found.")

        # Determine the insight type for supplemental DB queries
        insight_type_val = selected_value or (search_value or "").strip()

        # Fetch narrative columns and sample narrative from DB
        try:
            narrative_cols = get_narrative_columns_for_insight(insight_type_val) if insight_type_val else []
        except Exception:
            narrative_cols = []

        try:
            sample_narrative = get_sample_narrative_for_insight(insight_type_val) if insight_type_val else None
        except Exception:
            sample_narrative = None

        sections = []
        for _, r in filtered.iterrows():
            row_dict = r.to_dict()
            sections.append(
                build_rule_formatted_sections(
                    row_dict,
                    narrative_columns=narrative_cols,
                    sample_narrative=sample_narrative,
                )
            )
            sections.append(html.Hr(style={"margin": "8px 0", "borderColor": "#eee"}))

        if sections:
            sections = sections[:-1]

        return html.Div(sections, style={"display": "block", "width": "100%"})

