from dash import Dash, dcc, html, Input, Output, State, no_update
import urllib.parse
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from utils.data import get_metadata_transformed, get_rules_transformed, get_profile_tables, get_insight_details

# Import tab modules
from tabs.filter_tab import layout as filter_tab_layout, register_callbacks as register_filter_callbacks
from tabs.search_tab import layout as search_tab_layout, register_callbacks as register_search_callbacks
from tabs.rule_tab import layout as rule_tab_layout, register_callbacks as register_rule_callbacks
from tabs.profile_table_tab import layout as profile_table_tab_layout, register_callbacks as register_profile_table_callbacks
from tabs.column_search import layout as column_search_layout, register_callbacks as register_column_search_callbacks
from tabs.insight_details_tab import layout as insight_details_layout, register_callbacks as register_insight_details_callbacks
from tabs.table_lineage_tab import layout as table_lineage_tab_layout, register_callbacks as register_table_lineage_callbacks
from tabs.table_lineage_er_tab import (
    layout as table_lineage_er_tab_layout,
    register_callbacks as register_table_lineage_er_callbacks,
    register_server_routes as register_table_lineage_er_server_routes,
)

# ── New lineage tabs ──────────────────────────────────────────────────────────
from tabs.column_lineage_tab import (
    layout as column_lineage_tab_layout,
    register_callbacks as register_column_lineage_callbacks,
    register_server_routes as register_column_lineage_server_routes,
)
from tabs.insight_lineage_tab import (
    layout as insight_lineage_tab_layout,
    register_callbacks as register_insight_lineage_callbacks,
    register_server_routes as register_insight_lineage_server_routes,
)

# creating dash app
class InsightsMetaDashApp:
    def __init__(self,server):
        self.flask_server=server
        
        self.df = get_metadata_transformed()
        self.df_rules = get_rules_transformed()
        self.df_profile_tables = get_profile_tables()
        self.df_insight_details = get_insight_details()

        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Insights Metadata Dashboard",
            server=server,
            # Align client and server routes to avoid loading loop
            requests_pathname_prefix= "/insights_metadata/",
            routes_pathname_prefix= "/insights_metadata/",
        )
        
        routes_prefix = self.app.config.routes_pathname_prefix or "/"

        # Register Flask routes for iframe-served HTML tabs
        register_table_lineage_er_server_routes(server, routes_prefix)
        register_column_lineage_server_routes(server, routes_prefix)
        register_insight_lineage_server_routes(server, routes_prefix)

        # Enable extra Cytoscape layouts like 'cose-bilkent'
        try:
            cyto.load_extra_layouts()
        except Exception:
            # Safe to continue even if loading fails; falls back to built-in layouts
            pass


        # main app layout using modular tabs
        self.app.layout = html.Div(
            [
                dcc.Location(id="url"),
                html.Div(
                    [
                        html.H2(
                            "Insights Metadata Dashboard",
                            style={
                                "fontFamily": '"Inter", "Segoe UI", Arial, sans-serif',
                                "fontWeight": "800",
                                "fontSize": "1.6rem",
                                "color": "#2d3748",
                                "margin": "0",
                                "letterSpacing": "-0.01em",
                            },
                        ),
                    ],
                    style={
                        "padding": "16px 28px 12px",
                        "borderBottom": "3px solid #3182ce",
                        "background": "linear-gradient(135deg, #ffffff 0%, #ebf4ff 100%)",
                    },
                ),
                dcc.Tabs(
                    id="tabs-main",
                    value="tab-filter",
                    children=[
                        filter_tab_layout(self.df),
                        search_tab_layout(self.df),
                        column_search_layout(self.df),
                        rule_tab_layout(self.df_rules),
                        insight_details_layout(self.df_insight_details),
                        table_lineage_er_tab_layout(self.df_profile_tables),
                        # Uncomment the line below to add back table lineage horizontal tab
                        # profile_table_tab_layout(self.df_profile_tables), 
                        table_lineage_tab_layout(self.df_profile_tables),
                        column_lineage_tab_layout(),
                        insight_lineage_tab_layout(),
                    ],
                    style={
                        "marginBottom": "0",
                        "fontFamily": '"Inter", "Segoe UI", Arial, sans-serif',
                    },
                    colors={
                        "border": "#e2e8f0",
                        "primary": "#3182ce",
                        "background": "#f0f2f5",
                    },
                ),
            ],
            style={
                "fontFamily": '"Inter", "Segoe UI", Arial, sans-serif',
                "width": "100%",
                "overflowX": "hidden",
                "backgroundColor": "#f0f2f5",
                "minHeight": "100vh",
            },
        )

        # Navigate to third tab when an insight is present in URL search
        @self.app.callback(
            Output("tabs-main", "value"),
            Output("rule-search", "value"),
            Output("rule-search", "search_value"),
            Input("url", "search"),
            State("tabs-main", "value"),
            #prevent_initial_call=True,
        )
        def navigate_to_rule_tab(search, current_tab):
            if not search:
                return current_tab, no_update, no_update
            try:
                query = urllib.parse.parse_qs(search.lstrip("?"))
                insight = query.get("insight", [None])[0]
            except Exception:
                insight = None
            if not insight:
                return current_tab, no_update, no_update
            # Switch to rule tab and set the dropdown value and typed search
            return "tab-rule", insight, insight

        # Register tab-specific callbacks
        register_filter_callbacks(self.app, self.df)
        register_search_callbacks(self.app, self.df)
        register_column_search_callbacks(self.app, self.df)
        register_rule_callbacks(self.app, self.df_rules)
        register_insight_details_callbacks(self.app, self.df_insight_details)
        register_profile_table_callbacks(self.app)
        register_table_lineage_callbacks(self.app)
        register_table_lineage_er_callbacks(self.app)
        register_column_lineage_callbacks(self.app)
        register_insight_lineage_callbacks(self.app)
