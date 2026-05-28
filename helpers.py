from dash import html, dcc, no_update
import dash_bootstrap_components as dbc
import urllib.parse
import uuid

from styles import style as ms_style, ubs_colors, schema_info_styles, metadata_dashboard_styles as mds

# Label mapping for modal display
MODAL_LABEL_MAP = {
    "Table Name": "Profile Table Name",
    "odm_insight_types": "Attribute utilized in ODM rule and insight generation",
    "odm_target_type": "ODM Rule Target Type",
    "nlg_insight_types": "Attribute utilized in insight narrative",
    "nlg_target_type": "Insight Rule Target Type",
}

RULE_LABEL_MAP = {
    "target_type": "Target Type",
    "insight_type": "Insight Rule",
    "filename": "Rule File",
    "file_path": "Rule Path",
    "metric_name": "Metric Name",
    "metric_type": "Metric Type",
    "metric_val": "Metric Value",
    "insight_att_val": "Insight Attribute Name",
    "profile_table": "Profile Table",
    "rule_column": "Columns used in insight rule",
    "target_key": "Profile Key",
    "logic": "Rule Logic",
    "is_active": "Rule Active",
    "metric_1": "Attribution Metrics",
    "time_horizon_in_weeks": "Time Horizon (weeks)",
    "insight_eval_type": "Insight Evaluation Type",
}


def build_kv_section(
    section_title: str,
    payload: dict,
    label_map: dict | None = None,
    hyperlink_fields: tuple = ("odm_insight_types", "nlg_insight_types"),
    bulletize_comma_values: bool = False,
    highlight_yes_field=None,
):
    try:
        keys = list(payload.keys())
    except Exception:
        keys = []
    if ms_style:
        label_map = label_map or MODAL_LABEL_MAP
        _LABEL_STYLE = {
            "fontWeight": "600",
            "color": "#4a5568",
            "fontSize": "12px",
            "width": "280px",
            "minWidth": "280px",
            "flexShrink": "0",
            "paddingRight": "12px",
            "lineHeight": "1.5",
            "whiteSpace": "normal",
        }
        _VALUE_STYLE = {
            "color": "#1a202c",
            "fontSize": "13px",
            "lineHeight": "1.5",
            "wordBreak": "break-word",
        }
        _YES_STYLE = {
            **_VALUE_STYLE,
            "color": ubs_colors["Bordeaux I"]["RGB_py"],
            "fontWeight": "700",
        }
        rows = []
        for k in keys:
            value = "" if (payload.get(k) is None) else str(payload.get(k))
            display_label = label_map.get(str(k), str(k))

            is_highlighted = (
                highlight_yes_field
                and str(k) == str(highlight_yes_field)
                and str(value).strip().lower() == "yes"
            )
            val_style = _YES_STYLE if is_highlighted else _VALUE_STYLE

            if str(k) in hyperlink_fields and value and str(value).strip().lower() != "none":
                parts = [p.strip() for p in str(value).split(",") if p.strip() and p.strip().lower() != "none"]
                if len(parts) > 1:
                    value_comp = html.Ul(
                        [html.Li(dcc.Link(p, href=f"/?insight={urllib.parse.quote(p)}&rid={uuid.uuid4()}", style=_VALUE_STYLE)) for p in parts],
                        style={"margin": "0 0 0 14px", "padding": "0"},
                    )
                elif len(parts) == 1:
                    value_comp = dcc.Link(parts[0], href=f"/?insight={urllib.parse.quote(parts[0])}&rid={uuid.uuid4()}", style=_VALUE_STYLE)
                else:
                    value_comp = html.Span("—", style=val_style)
            elif bulletize_comma_values and "," in value:
                parts = [p.strip() for p in value.split(",") if p.strip()]
                value_comp = html.Ul(
                    [html.Li(p, style={"fontSize": "13px"}) for p in parts],
                    style={"margin": "0 0 0 14px", "padding": "0"},
                )
            else:
                value_comp = html.Span(value if value else "—", style=val_style)

            rows.append(
                html.Div(
                    [
                        html.Span(display_label, style=_LABEL_STYLE),
                        html.Div(value_comp, style={"flex": "1"}),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "flex-start",
                        "padding": "7px 16px",
                        "borderBottom": "1px solid #f7fafc",
                    },
                )
            )

        return dbc.Container(
            [
                html.Br(),
                dbc.Row(
                    [html.H5(section_title, style={"margin": "0", "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.04em", "textTransform": "uppercase"})],
                    style={
                        **ms_style.get("genera_info_column_title", {}),
                        "background": "linear-gradient(135deg, #2d3748 0%, #1a202c 100%)",
                        "borderRadius": "6px 6px 0 0",
                        "padding": "10px 16px",
                        "margin": "0",
                    },
                ),
                html.Div(
                    rows,
                    style={
                        "backgroundColor": "#ffffff",
                        "borderRadius": "0 0 6px 6px",
                        "border": "1px solid #e2e8f0",
                        "borderTop": "none",
                        "paddingBottom": "4px",
                    },
                ),
            ],
            fluid=True,
        )


def build_kv_modal_body(section_title: str, payload: dict):
    return build_kv_section(
        section_title=section_title,
        payload=payload,
        label_map=MODAL_LABEL_MAP,
        hyperlink_fields=("odm_insight_types", "nlg_insight_types"),
        bulletize_comma_values=False,
        highlight_yes_field="Contains CID",
    )


def build_dtype_mapping_table(payload: dict):
    rows = [
        ("Data Type",         payload.get("Data Type")),
        ("Max Length",        payload.get("max_length")),
        ("Numeric Precision", payload.get("numeric_precision")),
        ("Numeric Scale",     payload.get("numeric_scale")),
    ]
    body_rows = [
        html.Tr([
            html.Td(lbl, style={"fontWeight": "600", "color": "#4a5568", "fontSize": "12px", "whiteSpace": "nowrap", "paddingRight": "12px", "border": "none", "padding": "5px 12px"}),
            html.Td("—" if (v is None or str(v).strip() == "") else str(v), style={"color": "#1a202c", "fontSize": "12px", "border": "none", "padding": "5px 12px"}),
        ], style={"borderBottom": "1px solid #f0f2f5"})
        for lbl, v in rows
    ]
    return dbc.Container(
        [
            html.Br(),
            dbc.Row(
                [html.H5("Data Type Info", style={"margin": "0", "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.04em", "textTransform": "uppercase"})],
                style={
                    **ms_style.get("genera_info_column_title", {}),
                    "background": "linear-gradient(135deg, #2d3748 0%, #1a202c 100%)",
                    "borderRadius": "6px 6px 0 0",
                    "padding": "10px 16px",
                },
            ),
            dbc.Row(
                [html.Table(body_rows, style={"width": "100%", "borderCollapse": "collapse"})],
                style={
                    "padding": "8px 4px",
                    "backgroundColor": "#ffffff",
                    "borderRadius": "0 0 6px 6px",
                    "border": "1px solid #e2e8f0",
                    "borderTop": "none",
                },
            ),
        ],
        fluid=True,
    )


def build_rule_formatted_sections(rule_row: dict, narrative_columns: list = None, sample_narrative: str = None):
    """
    Build formatted sections for a rule row.

    Parameters
    ----------
    rule_row : dict
        The rule data row.
    narrative_columns : list, optional
        Columns used in narrative (from ikg_dictionary_metadata_auto_refresh query).
        Each entry will be rendered as a hyperlink to the Column Insights tab.
    sample_narrative : str, optional
        The sample narrative text. If None or empty, displays "N/A".
    """
    def get_val(candidates):
        for k in candidates:
            if k in rule_row:
                return rule_row.get(k)
            for rk in rule_row.keys():
                if str(rk).lower() == str(k).lower():
                    return rule_row.get(rk)
        return None

    _LINK_STYLE = {
        "color": "#3182ce",
        "textDecoration": "underline",
        "cursor": "pointer",
        "fontSize": "13px",
    }

    def _col_link(col_name):
        """Return a dcc.Link that navigates to the Column Insights tab."""
        return dcc.Link(
            col_name,
            href=f"/?col={urllib.parse.quote(str(col_name))}&rid={uuid.uuid4()}",
            style=_LINK_STYLE,
        )

    def _col_links_component(columns):
        """Given a list of column name strings, build a bulleted hyperlink list."""
        if not columns:
            return html.Span("—", style=ms_style.get("general_info_bullet_point_text", {}))
        if len(columns) == 1:
            return _col_link(columns[0])
        return html.Ul(
            [html.Li(_col_link(c)) for c in columns],
            style={"margin": "4px 0 8px 16px"},
        )

    sections = [
        (
            "Insight",
            [
                ("target_type", ["target_type", "odm_target_type", "nlg_target_type"]),
                ("insight_type", ["insight_type", "insight"]),
                ("filename", ["filename", "rule_file", "file_name"]),
                ("file_path", ["file_path", "rule_path", "path"]),
            ],
        ),
        (
            "Metric and Attribute Details",
            [
                ("metric_name", ["metric_name", "metric"]),
                ("metric_type", ["metric_type", "type"]),
                ("metric_val", ["metric_val", "metric_value", "value"]),
                ("insight_att_val", ["insight_att_val", "insight_attribute", "attribute"]),
            ],
        ),
        (
            "Attribution Framework",
            [
                ("metric_1", ["metric_1", "metric_one"]),
                ("time_horizon_in_weeks", ["time_horizon_in_weeks", "time_horizon"]),
                ("insight_eval_type", ["insight_eval_type", "insight_evaluation_type)"]),
             
            ],
        ),
        (
            "Profile Details",
            [
                ("profile_table", ["profile_table", "table_name", "profile_table_name"]),
                ("target_key", ["target_key", "profile_key", "key"]),
                ("rule_column", ["rule_column", "column_name", "profile_column"]),
            ],
        ),
        (
            "Rule Logic",
            [
                ("logic", ["logic", "rule_logic"]),
                ("is_active", ["is_active", "active"]),
            ],
        ),
    ]

    title_style = {
        **ms_style.get("genera_info_column_title", {}),
        "background": "linear-gradient(135deg, #2d3748 0%, #1a202c 100%)",
        "borderRadius": "6px 6px 0 0",
        "padding": "10px 16px",
    }
    text_style = {
        **ms_style.get("genera_info_column_text", {}),
        "padding": "12px 16px",
        "backgroundColor": "#ffffff",
        "borderRadius": "0 0 6px 6px",
        "border": "1px solid #e2e8f0",
        "borderTop": "none",
    }
    bullet_title_style = ms_style.get("general_info_bullet_point_title", {})
    bullet_text_style = ms_style.get("general_info_bullet_point_text", {})

    def render_section(title, items):
        children = []
        for key, candidates in items:
            label = RULE_LABEL_MAP.get(key, key)
            val = get_val(candidates)
            display_val = "" if val is None else str(val)

            if key == "rule_column":
                # Make each column a hyperlink to Column Insights tab
                parts = [p.strip() for p in display_val.split(",") if p.strip()] if display_val else []
                value_comp = _col_links_component(parts) if parts else html.Span("—", style=bullet_text_style)
            else:
                value_comp = html.Span(display_val if display_val else "—", style=bullet_text_style)

            children.extend([
                html.Span(f"{label}: ", style=bullet_title_style),
                value_comp,
                html.Br(),
            ])

        # Inject "Columns used in narrative" into Profile Details section
        if title == "Profile Details":
            narr_cols = narrative_columns or []
            children.extend([
                html.Span("Column used in narrative: ", style=bullet_title_style),
                _col_links_component(narr_cols),
                html.Br(),
            ])

        return dbc.Container(
            [
                html.Br(),
                dbc.Row([html.H5(title)], style=title_style),
                dbc.Row([html.Div(children)], style=text_style),
            ],
            fluid=True,
        )

    def render_sample_narrative_section():
        """Render the Sample Narrative section."""
        narr_text = sample_narrative if (sample_narrative and str(sample_narrative).strip() not in ("", "None", "none", "null")) else None
        content = html.Span(
            narr_text if narr_text else "N/A",
            style={
                **bullet_text_style,
                **({"fontStyle": "italic", "color": "#718096"} if not narr_text else {}),
            },
        )
        return dbc.Container(
            [
                html.Br(),
                dbc.Row([html.H5("Sample Narrative")], style=title_style),
                dbc.Row(
                    [html.Div([content], style={"padding": "12px 16px", "lineHeight": "1.7"})],
                    style=text_style,
                ),
            ],
            fluid=True,
        )

    out = []
    for title, items in sections:
        out.append(render_section(title, items))
        out.append(html.Hr(style={"margin": "8px 0", "borderColor": "#eee"}))

    # Append Sample Narrative section
    out.append(render_sample_narrative_section())

    return html.Div(out, style={"display": "block", "width": "100%"})


def display_modal_common(
    df,
    mode: str,
    active_cell,
    close_clicks,
    is_open,
    table_data,
    selected_col_value=None,
    typed_search_value=None,
):
    if active_cell:
        row_idx = active_cell.get("row")
        if row_idx is None or row_idx >= len(table_data or []):
            return is_open, "Details", html.P("No data available for selection."), None, [], []

        row = table_data[row_idx]
        table_name_val = row.get("Table Name")

        col_fields = [
            "Column Name",
            "Table Name",
            "Business Attribute Name",
            "Description",
            "Contains CID",
        ]
        rule_fields = ["odm_insight_types", "nlg_insight_types", "nlg_target_type"]

        sections = []
        if mode == "search":
            col_query = selected_col_value or (typed_search_value or "").strip()

            def _badge(label, value, color="#3182ce"):
                return html.Div(
                    [
                        html.Span(label, style={"fontSize": "0.6rem", "fontWeight": "700", "letterSpacing": "0.1em", "textTransform": "uppercase", "opacity": "0.7", "display": "block", "lineHeight": "1.2"}),
                        html.Span(str(value) if value else "—", style={"fontSize": "0.9rem", "fontWeight": "600", "lineHeight": "1.3", "display": "block"}),
                    ],
                    style={"backgroundColor": "rgba(255,255,255,0.13)", "borderLeft": f"3px solid {color}", "borderRadius": "5px", "padding": "4px 12px", "display": "inline-block"},
                )
            if selected_col_value:
                modal_title = html.Div([_badge("Column", selected_col_value, "#63b3ed"), _badge("Table", table_name_val, "#68d391")], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "center"})
            elif col_query:
                modal_title = html.Div([_badge("Table", table_name_val, "#68d391"), _badge("Query", col_query, "#f6ad55")], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "center"})
            else:
                modal_title = html.Div([_badge("Table", table_name_val, "#68d391")], style={"display": "flex", "gap": "10px", "alignItems": "center"})

            full_rows = None
            try:
                if table_name_val is not None and col_query:
                    if selected_col_value:
                        match = df[
                            (df["Table Name"].astype(str) == str(table_name_val))
                            & (df["Column Name"].astype(str) == str(selected_col_value))
                        ]
                    else:
                        sv = str(col_query)
                        match = df[
                            (df["Table Name"].astype(str) == str(table_name_val))
                            & (
                                df["Column Name"].astype(str).str.contains(
                                    sv, case=False, na=False
                                )
                            )
                        ]
                    if not match.empty:
                        full_rows = match
            except Exception:
                full_rows = None

            if full_rows is None or full_rows.empty:
                details_body = html.P(
                    "No matching column details found for this table."
                )
            else:
                for _, r in full_rows.iterrows():
                    payload = r.to_dict()
                    col_section = {k: payload.get(k) for k in col_fields}
                    rules_section = {k: payload.get(k) for k in rule_fields}
                    sections.append(
                        dbc.Row(
                            [
                                dbc.Col(
                                    build_kv_modal_body("Column Details:", col_section),
                                    width=8,
                                ),
                                dbc.Col(
                                    build_dtype_mapping_table(payload), width=4
                                ),
                            ],
                            style={"marginBottom": "8px"},
                        )
                    )
                    sections.append(
                        html.Hr(style={"margin": "8px 0", "borderColor": "#eee"})
                    )
                    sections.append(
                        build_kv_section(
                            section_title="STAAT Insight Rules:",
                            payload=rules_section,
                            label_map=MODAL_LABEL_MAP,
                            bulletize_comma_values=True,
                        )
                    )
                    sections.append(
                        html.Hr(style={"margin": "8px 0", "borderColor": "#eee"})
                    )
                details_body = html.Div(
                    sections, style={"display": "block", "width": "100%"}
                )

            return True, modal_title, details_body, None, [], []

        column_name_val = row.get("Column Name")
        def _badge(label, value, color="#3182ce"):
            return html.Div(
                [
                    html.Span(label, style={"fontSize": "0.6rem", "fontWeight": "700", "letterSpacing": "0.1em", "textTransform": "uppercase", "opacity": "0.7", "display": "block", "lineHeight": "1.2"}),
                    html.Span(str(value) if value else "—", style={"fontSize": "0.9rem", "fontWeight": "600", "lineHeight": "1.3", "display": "block"}),
                ],
                style={"backgroundColor": "rgba(255,255,255,0.13)", "borderLeft": f"3px solid {color}", "borderRadius": "5px", "padding": "4px 12px", "display": "inline-block"},
            )
        modal_title = html.Div(
            [_badge("Column", column_name_val, "#63b3ed"), _badge("Table", table_name_val, "#68d391")],
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "center"},
        )

        full_row_dict = None
        try:
            if (
                table_name_val is not None
                and column_name_val is not None
                and ("Table Name" in df.columns)
                and ("Column Name" in df.columns)
            ):
                match = df[
                    (df["Table Name"].astype(str) == str(table_name_val))
                    & (df["Column Name"].astype(str) == str(column_name_val))
                ]
                if not match.empty:
                    full_row_dict = match.iloc[0].to_dict()
        except Exception:
            full_row_dict = None

        payload = (full_row_dict or row)
        col_section = {k: payload.get(k) for k in col_fields}
        rules_section = {k: payload.get(k) for k in rule_fields}

        details_body = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            build_kv_modal_body("Column Details:", col_section),
                            width=8,
                        ),
                        dbc.Col(build_dtype_mapping_table(payload), width=4),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Hr(style={"margin": "8px 0", "borderColor": "#eee"}),
                build_kv_section(
                    section_title="STAAT Insight Rules:",
                    payload=rules_section,
                    label_map=MODAL_LABEL_MAP,
                    bulletize_comma_values=True,
                ),
            ],
            style={"display": "block", "width": "100%"},
        )

        return True, modal_title, details_body, None, [], []

    if close_clicks:
        return False, no_update, no_update, None, [], []

    if is_open is False:
        return False, no_update, no_update, None, [], []

    return is_open, no_update, no_update, no_update, no_update, no_update

def summary_card_profile(
    table_name: str,
    total_columns: int,
    cid_columns: int
    # nlg_impacted: int,
    # odm_impacted: int,
):
    def _stat_card(title, value, card_style_key="stat_card_style", value_style_key="stat_card_value_style"):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(title, style=mds["stat_card_title_style"]),
                    html.Div(str(value), style=mds[value_style_key]),
                ],
                style=mds["stat_card_body_style"],
            ),
            style=mds[card_style_key],
        )

    return html.Div(
        [
            html.Div(
                table_name,
                style={
                    **mds["sub_header_style"],
                    "fontSize": "0.95rem",
                    "marginBottom": "12px",
                    "wordBreak": "break-all",
                },
            ),
            html.Div(
                [
                    _stat_card("Total Columns", total_columns, "stat_card_style", "stat_card_value_style"),
                    _stat_card("Contain CID", cid_columns, "stat_card_alert_style", "stat_card_value_alert_style"),
                    # _stat_card("Narrative Impact", nlg_impacted, "stat_card_green_style", "stat_card_value_green_style"),
                    # _stat_card("ODM Rule Impact", odm_impacted, "stat_card_orange_style", "stat_card_value_orange_style"),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "12px"},
            ),
        ],
        style={
            **mds["content_card_style"],
            "marginBottom": "12px",
        },
    )
