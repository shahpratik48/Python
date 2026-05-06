from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import uuid
import pandas as pd
from utils.data import (
    get_insight_release,
    get_odm_release_details,
    get_prev_closed_issue_ids,
    apply_insight_release_changes,
    apply_summary_release_changes,
    apply_exclusion_changes,
    get_swat_release,
    get_exclusion_rules,
)

from tabs.insight_release_tab import (
    layout as insight_release_layout,
    build_insight_display,
    build_tooltip_data,
    compute_insight_stats,
    format_count,
    format_weight_value,
)
from tabs.odm_release_tab import (
    layout as odm_release_layout,
    build_odm_display,
    build_odm_tooltip_data,
    compute_odm_stats,
    format_count as odm_format_count,
)
from tabs.swat_ticket_tracking_tab import layout as swat_tracking_layout, build_swat_display
from tabs.swat_summary_stats_tab import layout as swat_summary_layout, build_summary_table, build_insights_summary_table
from tabs.release_summary_tab import (
    layout as release_summary_layout,
    build_release_summary_display,
    compute_release_summary_stats,
)
from tabs.insight_exclusion_tab import layout as insight_exclusion_layout
from tabs.insight_throughput_tab import (
    layout as insight_throughput_layout,
    build_throughput_display,
    build_pipeline_cards,
    compute_ikg_only_count,
    compute_odm_closed_counts,
    compute_open_current_iteration_count,
)
from tabs.executive_summary_tab import (
    layout as executive_summary_layout,
    get_exec_summary_data,
    get_month_options_from_db,
    generate_summary,
    build_save_content,
)



class InsightsReleaseDashApp:
    def __init__(self,server):
        self.flask_server=server
        self.df = get_insight_release()
        self.df_odm = get_odm_release_details()
        self.df_prev_closed = get_prev_closed_issue_ids()
        self.df_swat = get_swat_release()
        self.df_exclusion = get_exclusion_rules()
        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Insights Release Dashboard",
            server=server,
            # Align client and server routes to avoid loading loop
            requests_pathname_prefix= "/insights_release/",
            routes_pathname_prefix= "/insights_release/",
        )

        self.app.layout = html.Div(
                [
                    dcc.Location(id="url"),
                    dcc.Interval(
                        id="refresh-interval",
                        interval= 30 * 60 * 1000, # refresh every 30 minutes (in miliseconds)
                        n_intervals=0,
                    ),
                    dcc.Tabs(
                        id="tabs-main",
                        value="tab-insight-details",
                        children=[
                            insight_release_layout(self.df, self.df_prev_closed),
                            release_summary_layout(self.df),
                            odm_release_layout(self.df_odm),
                            swat_tracking_layout(self.df_swat),
                            swat_summary_layout(self.df_swat),
                            insight_exclusion_layout(self.df_exclusion),
                            insight_throughput_layout(self.df_swat, self.df, self.df_odm),
                            executive_summary_layout(self.df),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                ],
                style={"fontFamily": "Segoe UI", "width": "100%", "overflowX": "hidden"},
            )

        @self.app.callback(
            Output("insight-release-details-store", "data", allow_duplicate=True),
            Output("insight-release-details-table", "data", allow_duplicate=True),
            Output("insight-release-iteration-filter", "options"),
            Output("insight-release-iteration-filter", "value", allow_duplicate=True),
            Output("release-summary-details-store", "data"),
            Output("release-summary-details-table", "data", allow_duplicate=True),
            Output("release-summary-iteration-filter", "options"),
            Output("release-summary-iteration-filter", "value", allow_duplicate=True),
            Output("odm-release-details-store", "data"),
            Output("odm-release-details-table", "data", allow_duplicate=True),
            Output("odm-release-iteration-end-date-filter", "options"),
            Output("swat-tracking-store", "data"),
            Output("swat-tracking-table", "data", allow_duplicate=True),
            Output("swat-summary-raw-store", "data"),
            Output("swat-summary-table", "data", allow_duplicate=True),
            Output("swat-summary-table", "columns", allow_duplicate=True),
            Output("swat-summary-insights-table", "data", allow_duplicate=True),
            Output("swat-summary-insights-table", "columns", allow_duplicate=True),
            Output("swat-summary-year-filter", "options"),
            Output("insight-exclusion-store", "data", allow_duplicate=True),
            Output("insight-exclusion-table", "data", allow_duplicate=True),
            Output("throughput-raw-store", "data", allow_duplicate=True),
            Output("throughput-insight-store", "data", allow_duplicate=True),
            Output("throughput-odm-store", "data", allow_duplicate=True),
            Output("throughput-pipeline-container", "children", allow_duplicate=True),
            #Input("url", "href"),
            Input("refresh-interval", "n_intervals"),
            State("insight-release-iteration-filter", "value"),
            State("release-summary-iteration-filter", "value"),
            State("odm-release-iteration-end-date-filter", "value"),
            State("odm-release-go-live-date-range", "start_date"),
            State("odm-release-go-live-date-range", "end_date"),
            prevent_initial_call="initial_duplicate",
        )
        def refresh_data(
            #_url_href,
            _n_intervals,
            insight_iteration_filter,
            summary_iteration_filter,
            odm_iteration_filter,
            odm_start_date,
            odm_end_date,
        ):
            latest_insight = get_insight_release()
            latest_odm = get_odm_release_details()
            latest_prev_closed = get_prev_closed_issue_ids()
            latest_swat = get_swat_release()
            latest_exclusion = get_exclusion_rules()

            insight_display, insight_options = build_insight_display(
                latest_insight, latest_prev_closed
            )
            summary_display, summary_options = build_release_summary_display(latest_insight)
            odm_display, odm_options = build_odm_display(latest_odm)
            swat_display = build_swat_display(latest_swat)

            insight_rows = insight_display.to_dict("records")
            summary_rows = summary_display.to_dict("records")
            odm_rows = odm_display.to_dict("records")

            # Auto-advance filter to new max date when one appears
            insight_new_max = max((opt["value"] for opt in insight_options), default=None)
            if insight_new_max and (not insight_iteration_filter or insight_new_max > str(insight_iteration_filter)):
                effective_insight_filter = insight_new_max
            else:
                effective_insight_filter = insight_iteration_filter

            summary_new_max = max((opt["value"] for opt in summary_options), default=None)
            if summary_new_max and (not summary_iteration_filter or summary_new_max > str(summary_iteration_filter)):
                effective_summary_filter = summary_new_max
            else:
                effective_summary_filter = summary_iteration_filter

            if effective_insight_filter:
                insight_selected = str(effective_insight_filter)
                insight_rows = [
                    row
                    for row in insight_rows
                    if str(row.get("iteration_end_date", "")) == insight_selected
                ]

            if effective_summary_filter:
                summary_selected = str(effective_summary_filter)
                summary_rows = [
                    row
                    for row in summary_rows
                    if str(row.get("iteration_end_date", "")) == summary_selected
                ]

            if odm_iteration_filter:
                odm_selected = str(odm_iteration_filter)
                odm_rows = [
                    row
                    for row in odm_rows
                    if str(row.get("iteration_end_date", "")) == odm_selected
                ]

            if odm_start_date or odm_end_date:
                start_dt = pd.to_datetime(odm_start_date).date() if odm_start_date else None
                end_dt = pd.to_datetime(odm_end_date).date() if odm_end_date else None

                def in_range(value):
                    if value in (None, ""):
                        return False
                    date_val = pd.to_datetime(value, errors="coerce")
                    if pd.isna(date_val):
                        return False
                    date_val = date_val.date()
                    if start_dt and date_val < start_dt:
                        return False
                    if end_dt and date_val > end_dt:
                        return False
                    return True

                odm_rows = [row for row in odm_rows if in_range(row.get("go_live_date"))]

            swat_rows = swat_display.to_dict("records")

            # Rebuild the SWAT summary pivots
            swat_summary_pivot, swat_year_opts, _ = build_summary_table(latest_swat)
            swat_summary_rows = swat_summary_pivot.to_dict("records")
            swat_summary_cols = [{"name": c, "id": c} for c in swat_summary_pivot.columns]

            swat_insights_pivot, _opts2, _ = build_insights_summary_table(latest_swat)
            swat_insights_rows = swat_insights_pivot.to_dict("records")
            swat_insights_cols = [{"name": c, "id": c} for c in swat_insights_pivot.columns]

            exclusion_records = latest_exclusion.to_dict("records")

            throughput_counts = build_throughput_display(latest_swat)
            ikg_only_count = compute_ikg_only_count(latest_insight)
            on_hold_count, live_count = compute_odm_closed_counts(latest_odm)
            open_current_iter_count = compute_open_current_iteration_count(latest_insight, latest_swat)
            throughput_cards = build_pipeline_cards(throughput_counts, ikg_only_count, on_hold_count, live_count, open_current_iter_count)

            return (
                insight_display.to_dict("records"),
                insight_rows,
                insight_options,
                effective_insight_filter,
                summary_display.to_dict("records"),
                summary_rows,
                summary_options,
                effective_summary_filter,
                odm_display.to_dict("records"),
                odm_rows,
                odm_options,
                swat_rows,
                swat_rows,
                latest_swat.to_dict("records"),
                swat_summary_rows,
                swat_summary_cols,
                swat_insights_rows,
                swat_insights_cols,
                swat_year_opts,
                exclusion_records,
                exclusion_records,
                latest_swat.to_dict("records"),
                latest_insight.to_dict("records"),
                latest_odm.to_dict("records"),
                throughput_cards,
            )

        @self.app.callback(
            Output("insight-release-details-table", "data"),
            Output("insight-release-details-table", "tooltip_data"),
            Input("insight-release-iteration-filter", "value"),
            State("insight-release-details-store", "data"),
        )
        def filter_by_iteration_end_date(selected_date, original_rows):
            original_rows = original_rows or []
            if not selected_date:
                filtered = original_rows
            else:
                selected_value = str(selected_date)
                filtered = [
                    row
                    for row in original_rows
                    if str(row.get("iteration_end_date", "")) == selected_value
                ]
            return filtered, build_tooltip_data(filtered)

        @self.app.callback(
            Output("insight-release-stat-distinct-id", "children"),
            Output("insight-release-stat-new-insights", "children"),
            Output("insight-release-stat-total-weight", "children"),
            Output("insight-release-stat-swat-count", "children"),
            Output("insight-release-stat-cid-count", "children"),
            Output("insight-release-stat-ikg-no", "children"),
            Output("insight-release-stat-nlg-no", "children"),
            Output("insight-release-stat-odm-no", "children"),
            Input("insight-release-details-table", "data"),
        )
        def update_insight_stats(rows):
            stats = compute_insight_stats(rows or [])
            return (
                format_count(stats["distinct_id_count"]),
                format_count(stats["new_insights_count"]),
                format_weight_value(stats["total_weight"]),
                format_count(stats["swat_count"]),
                format_count(stats["cid_count"]),
                format_count(stats["ikg_no_count"]),
                format_count(stats["nlg_no_count"]),
                format_count(stats["odm_no_count"]),
            )

        @self.app.callback(
            Output("insight-issue-summary-modal", "is_open"),
            Output("insight-issue-summary-modal-content", "children"),
            Input("insight-release-details-table", "active_cell"),
            Input("insight-issue-summary-modal-close", "n_clicks"),
            State("insight-release-details-table", "derived_virtual_data"),
            prevent_initial_call=True,
        )
        def toggle_insight_issue_summary_modal(active_cell, _close_clicks, table_data):
            triggered_id = ctx.triggered_id
            if triggered_id == "insight-issue-summary-modal-close":
                return False, no_update
            if active_cell and active_cell.get("column_id") == "issue_summary":
                row_idx = active_cell.get("row", -1)
                table_data = table_data or []
                if 0 <= row_idx < len(table_data):
                    issue_summary = table_data[row_idx].get("issue_summary", "") or ""
                    if issue_summary.strip():
                        return True, issue_summary
            return no_update, no_update

        @self.app.callback(
            Output("odm-release-details-table", "data"),
            Output("odm-release-details-table", "tooltip_data"),
            Input("odm-release-iteration-end-date-filter", "value"),
            Input("odm-release-go-live-date-range", "start_date"),
            Input("odm-release-go-live-date-range", "end_date"),
            State("odm-release-details-store", "data"),
        )
        def filter_by_go_live_date(selected_iteration, start_date, end_date, original_rows):
            original_rows = original_rows or []
            filtered_rows = original_rows

            if selected_iteration:
                selected_value = str(selected_iteration)
                filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("iteration_end_date", "")) == selected_value
                ]

            if start_date or end_date:
                start_dt = pd.to_datetime(start_date).date() if start_date else None
                end_dt = pd.to_datetime(end_date).date() if end_date else None

                def in_range(value):
                    if value in (None, ""):
                        return False
                    date_val = pd.to_datetime(value, errors="coerce")
                    if pd.isna(date_val):
                        return False
                    date_val = date_val.date()
                    if start_dt and date_val < start_dt:
                        return False
                    if end_dt and date_val > end_dt:
                        return False
                    return True

                filtered_rows = [
                    row for row in filtered_rows if in_range(row.get("go_live_date"))
                ]

            return filtered_rows, build_odm_tooltip_data(filtered_rows)

        @self.app.callback(
            Output("odm-release-stat-new-insights", "children"),
            Output("odm-release-stat-issue-count", "children"),
            Output("odm-release-stat-on-hold", "children"),
            Output("odm-release-stat-upcoming", "children"),
            Input("odm-release-details-table", "data"),
        )
        def update_odm_stats(rows):
            stats = compute_odm_stats(rows or [])
            return (
                odm_format_count(stats["distinct_rule_name_count"]),
                odm_format_count(stats["distinct_issue_id_count"]),
                odm_format_count(stats["distinct_rule_name_on_hold_count"]),
                odm_format_count(stats["distinct_rule_name_upcoming_count"]),

            )

        @self.app.callback(
            Output("release-summary-details-table", "data"),
            Output("release-summary-details-table", "tooltip_data"),
            Input("release-summary-iteration-filter", "value"),
            State("release-summary-details-store", "data"),
        )
        def filter_release_summary_by_iteration_end_date(selected_date, original_rows):
            original_rows = original_rows or []
            if not selected_date:
                return original_rows, build_tooltip_data(original_rows)
            selected_value = str(selected_date)
            filtered = [
                row
                for row in original_rows
                if str(row.get("iteration_end_date", "")) == selected_value
            ]
            return filtered, build_tooltip_data(filtered)

        @self.app.callback(
            Output("release-summary-stat-distinct-id", "children"),
            Output("release-summary-stat-new-insights", "children"),
            Output("release-summary-stat-total-weight", "children"),
            Output("release-summary-stat-ikg-no", "children"),
            Output("release-summary-stat-nlg-no", "children"),
            Output("release-summary-stat-odm-no", "children"),
            Input("release-summary-details-table", "data"),
        )
        def update_release_summary_stats(rows):
            stats = compute_release_summary_stats(rows or [])
            return (
                format_count(stats["distinct_id_count"]),
                format_count(stats["new_insights_count"]),
                format_weight_value(stats["total_weight"]),
                format_count(stats["ikg_no_count"]),
                format_count(stats["nlg_no_count"]),
                format_count(stats["odm_no_count"]),
            )

        @self.app.callback(
            Output("insight-release-details-table", "data", allow_duplicate=True),
            Input("insight-release-add-row", "n_clicks"),
            State("insight-release-details-table", "data"),
            State("insight-release-details-table", "columns"),
            prevent_initial_call=True,
        )
        
        def add_insight_row(n_clicks, rows, columns):
            if not n_clicks:
                return no_update
            rows = rows or []
            new_row = {col["id"]: "" for col in (columns or [])}
            new_row["id"] = f"new-{uuid.uuid4()}"
            new_row["row_priority"] = 1
            rows.insert(0, new_row)
            return rows

        @self.app.callback(
            Output("insight-release-details-table", "data", allow_duplicate=True),
            Input("insight-release-clone-row", "n_clicks"),
            State("insight-release-details-table", "data"),
            State("insight-release-details-table", "selected_rows"),
            prevent_initial_call=True,
        )
        def clone_insight_row(n_clicks, rows, selected_rows):
            if not n_clicks:
                return no_update
            rows = rows or []
            selected_rows = selected_rows or []
            if not selected_rows:
                return no_update
            row_idx = selected_rows[0]
            if row_idx is None or row_idx < 0 or row_idx >= len(rows):
                return no_update

            source_row = rows[row_idx]
            cloned_row = dict(source_row)
            cloned_row["id"] = f"new-{uuid.uuid4()}"
            cloned_row["row_priority"] = 1
            rows.insert(0, cloned_row)
            return rows

        @self.app.callback(
            Output("insight-release-details-store", "data"),
            Output("insight-release-save-toast", "children"),
            Output("insight-release-save-toast", "is_open"),
            Input("insight-release-save", "n_clicks"),
            State("insight-release-details-table", "data"),
            State("insight-release-details-store", "data"),
            State("insight-release-iteration-filter", "value"),
            prevent_initial_call=True,
        )
        def save_insight_changes(n_clicks, current_rows, original_rows, iteration_filter):
            if not n_clicks:
                return no_update, no_update, False

            current_rows = current_rows or []
            original_rows = original_rows or []

            ignore_columns = {"id", "row_priority", "branch_name_invalid", "prev_closed_match"}

            def normalize(value):
                if value is None:
                    return ""
                return value

            def append_manual_label(row):
                if not isinstance(row, dict):
                    return False
                labels_value = row.get("labels", "")
                if labels_value is None:
                    labels_value = ""
                labels_str = str(labels_value).strip()
                manual_tag = "manually altered"
                if manual_tag in labels_str:
                    row["labels"] = labels_str
                    return False
                if labels_str:
                    row["labels"] = f"{labels_str}, {manual_tag}"
                else:
                    row["labels"] = manual_tag
                return True

            original_by_id = {
                row.get("id"): row for row in original_rows if row.get("id") is not None
            }
            current_by_id = {
                row.get("id"): row for row in current_rows if row.get("id") is not None
            }

            def _matches_filter(row):
                if not iteration_filter:
                    return True
                return str(row.get("iteration_end_date", "")) == str(iteration_filter)

            original_filtered_ids = {
                row_id
                for row_id, row in original_by_id.items()
                if _matches_filter(row)
            }
            current_filtered_ids = set(current_by_id.keys())

            deleted_ids = original_filtered_ids - current_filtered_ids

            merged_by_id = dict(original_by_id)
            for row_id, row in current_by_id.items():
                merged_by_id[row_id] = row
            for row_id in deleted_ids:
                merged_by_id.pop(row_id, None)

            original_ids = set(original_by_id.keys())
            merged_ids = set(merged_by_id.keys())

            added_ids = merged_ids - original_ids
            common_ids = original_ids & merged_ids

            added_rows = [merged_by_id[row_id] for row_id in added_ids]
            deleted_rows = [original_by_id[row_id] for row_id in deleted_ids]
            updated_rows = []

            for row_id in common_ids:
                original_row = original_by_id[row_id]
                current_row = merged_by_id[row_id]
                changed_columns = []
                for col in current_row.keys():
                    if col in ignore_columns:
                        continue
                    if normalize(current_row.get(col)) != normalize(original_row.get(col)):
                        changed_columns.append(col)
                if changed_columns:
                    if append_manual_label(current_row) and "labels" not in changed_columns:
                        changed_columns.append("labels")
                    updated_rows.append((original_row, current_row, changed_columns))

            def strip_internal_fields(row):
                return {
                    key: value
                    for key, value in row.items()
                    if key not in ignore_columns
                }

            if added_rows:
                for row in added_rows:
                    append_manual_label(row)

            if not added_rows and not updated_rows and not deleted_rows:
                return no_update, "No changes to save.", True

            try:
                cleaned_added_rows = [strip_internal_fields(row) for row in added_rows]
                cleaned_updated_rows = [
                    (strip_internal_fields(original), strip_internal_fields(current), cols)
                    for original, current, cols in updated_rows
                ]
                cleaned_deleted_rows = [strip_internal_fields(row) for row in deleted_rows]
                apply_insight_release_changes(
                    cleaned_added_rows,
                    cleaned_updated_rows,
                    cleaned_deleted_rows,
                )
            except Exception as exc:
                return no_update, f"Save failed: {exc}", True

            summary = (
                f"Saved {len(added_rows)} added, {len(updated_rows)} updated, {len(deleted_rows)} deleted rows."
            )
            return list(merged_by_id.values()), summary, True

        @self.app.callback(
            Output("release-summary-details-store", "data", allow_duplicate=True),
            Output("release-summary-save-toast", "children"),
            Output("release-summary-save-toast", "is_open"),
            Input("release-summary-save", "n_clicks"),
            State("release-summary-details-table", "data"),
            State("release-summary-details-store", "data"),
            State("release-summary-iteration-filter", "value"),
            prevent_initial_call=True,
        )
        def save_release_summary_changes(n_clicks, current_rows, original_rows, iteration_filter):
            if not n_clicks:
                return no_update, no_update, False

            EDITABLE_COLS = {"uat_comment", "uat_complete", "approved_for_release", "issue_author_name"}

            current_rows = current_rows or []
            original_rows = original_rows or []

            def normalize(value):
                if value is None:
                    return ""
                return str(value)

            original_by_id = {
                row.get("id"): row for row in original_rows if row.get("id") is not None
            }
            current_by_id = {
                row.get("id"): row for row in current_rows if row.get("id") is not None
            }

            updated_rows = []
            seen_id_x = set()

            for row_id, current_row in current_by_id.items():
                original_row = original_by_id.get(row_id)
                if original_row is None:
                    continue

                changed_columns = [
                    col for col in EDITABLE_COLS
                    if normalize(current_row.get(col)) != normalize(original_row.get(col))
                ]

                if changed_columns:
                    id_x = original_row.get("id_x")
                    if id_x in seen_id_x:
                        continue
                    seen_id_x.add(id_x)
                    updated_rows.append((original_row, current_row, changed_columns))

            if not updated_rows:
                return no_update, "No changes to save.", True

            try:
                apply_summary_release_changes(updated_rows)
            except Exception as exc:
                return no_update, f"Save failed: {exc}", True

            merged_by_id = dict(original_by_id)
            for row_id, row in current_by_id.items():
                merged_by_id[row_id] = row

            summary_msg = f"Saved {len(updated_rows)} updated row(s)."
            return list(merged_by_id.values()), summary_msg, True

        @self.app.callback(
            Output("insight-release-download", "data"),
            Input("insight-release-export", "n_clicks"),
            State("insight-release-details-table", "data"),
            State("insight-release-details-table", "columns"),
            prevent_initial_call=True,
        )
        def export_insight_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "insight_release.csv",
                index=False,
            )

        @self.app.callback(
            Output("release-summary-download", "data"),
            Input("release-summary-export", "n_clicks"),
            State("release-summary-details-table", "data"),
            State("release-summary-details-table", "columns"),
            prevent_initial_call=True,
        )
        def export_release_summary_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "release_summary.csv",
                index=False,
            )

        @self.app.callback(
            Output("odm-release-download", "data"),
            Input("odm-release-export", "n_clicks"),
            State("odm-release-details-table", "data"),
            State("odm-release-details-table", "columns"),
            prevent_initial_call=True,
        )
        def export_odm_release_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "odm_release.csv",
                index=False,
            )

        @self.app.callback(
            Output("swat-description-modal", "is_open"),
            Output("swat-description-modal-content", "children"),
            Input("swat-tracking-table", "active_cell"),
            Input("swat-description-modal-close", "n_clicks"),
            State("swat-tracking-table", "derived_virtual_data"),
            prevent_initial_call=True,
        )
        def toggle_swat_description_modal(active_cell, _close_clicks, table_data):
            triggered_id = ctx.triggered_id
            if triggered_id == "swat-description-modal-close":
                return False, no_update
            if active_cell and active_cell.get("column_id") == "issue_summary":
                row_idx = active_cell.get("row", -1)
                table_data = table_data or []
                if 0 <= row_idx < len(table_data):
                    description = table_data[row_idx].get("issue_summary", "") or ""
                    if description.strip():
                        return True, description
            return no_update, no_update

        @self.app.callback(
            Output("swat-tracking-download", "data"),
            Input("swat-tracking-export", "n_clicks"),
            State("swat-tracking-table", "data"),
            State("swat-tracking-table", "columns"),
            prevent_initial_call=True,
        )
        def export_swat_tracking_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "swat_ticket_tracking.csv",
                index=False,
            )

        # ── SWAT Summary Stats callbacks ──────────────────────────────

        @self.app.callback(
            Output("swat-summary-table", "data"),
            Output("swat-summary-table", "columns"),
            Output("swat-summary-insights-table", "data"),
            Output("swat-summary-insights-table", "columns"),
            Input("swat-summary-year-filter", "value"),
            State("swat-summary-raw-store", "data"),
            prevent_initial_call=True,
        )
        def filter_swat_summary_by_year(selected_year, raw_records):
            raw_records = raw_records or []
            df_raw = pd.DataFrame(raw_records)
            pivot, _opts, _yr = build_summary_table(df_raw, selected_year)
            cols = [{"name": c, "id": c} for c in pivot.columns]
            insights_pivot, _opts2, _yr2 = build_insights_summary_table(df_raw, selected_year)
            insights_cols = [{"name": c, "id": c} for c in insights_pivot.columns]
            return pivot.to_dict("records"), cols, insights_pivot.to_dict("records"), insights_cols

        @self.app.callback(
            Output("swat-summary-download", "data"),
            Input("swat-summary-export", "n_clicks"),
            State("swat-summary-table", "data"),
            State("swat-summary-table", "columns"),
            prevent_initial_call=True,
        )
        def export_swat_summary_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "swat_summary_stats.csv",
                index=False,
            )

        @self.app.callback(
            Output("swat-summary-insights-download", "data"),
            Input("swat-summary-insights-export", "n_clicks"),
            State("swat-summary-insights-table", "data"),
            State("swat-summary-insights-table", "columns"),
            prevent_initial_call=True,
        )
        def export_swat_summary_insights_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "swat_summary_insights.csv",
                index=False,
            )

        @self.app.callback(
            Output("throughput-download", "data"),
            Input("throughput-export", "n_clicks"),
            State("throughput-raw-store", "data"),
            State("throughput-insight-store", "data"),
            State("throughput-odm-store", "data"),
            prevent_initial_call=True,
        )
        def export_throughput_csv(n_clicks, raw_records, insight_records, odm_records):
            if not n_clicks:
                return no_update

            raw_records = raw_records or []
            df_raw = pd.DataFrame(raw_records)
            status_counts = build_throughput_display(df_raw)

            df_insight_raw = pd.DataFrame(insight_records or [])
            ikg_only_count = compute_ikg_only_count(df_insight_raw)

            df_odm_raw = pd.DataFrame(odm_records or [])
            on_hold_count, live_count = compute_odm_closed_counts(df_odm_raw)

            rows = [(s, c) for s, c in status_counts]
            rows.append(("IKG Only", ikg_only_count))
            rows.append(("Closed – Insights On Hold", on_hold_count))
            rows.append(("Closed – Insights Live", live_count))
            df = pd.DataFrame(rows, columns=["IKG Status", "Weight"])

            return dcc.send_data_frame(
                df.to_csv,
                "insight_throughput_tracking.csv",
                index=False,
            )

        @self.app.callback(
            Output("insight-exclusion-download", "data"),
            Input("insight-exclusion-export", "n_clicks"),
            State("insight-exclusion-table", "data"),
            State("insight-exclusion-table", "columns"),
            prevent_initial_call=True,
        )
        def export_insight_exclusion_csv(n_clicks, table_data, table_columns):
            if not n_clicks:
                return no_update

            table_data = table_data or []
            columns = [col.get("id") for col in (table_columns or []) if col.get("id")]
            df = pd.DataFrame(table_data)
            if columns:
                df = df[[col for col in columns if col in df.columns]]

            return dcc.send_data_frame(
                df.to_csv,
                "insight_exclusion.csv",
                index=False,
            )
        
        
        # ── Executive Summary callbacks ─────────────────────────────

        @self.app.callback(
            Output("exec-summary-month-filter", "options"),
            Input("refresh-interval", "n_intervals"),
            prevent_initial_call="initial_duplicate",
        )
        def refresh_exec_summary_data(_n_intervals):
            return get_month_options_from_db()

        @self.app.callback(
            Output("exec-summary-output", "children"),
            Output("exec-summary-text-store", "data"),
            Output("exec-summary-save-btn", "disabled"),
            Input("exec-summary-generate-btn", "n_clicks"),
            State("exec-summary-month-filter", "value"),
            prevent_initial_call=True,
        )
        def generate_exec_summary(n_clicks, selected_month):
            if not n_clicks or not selected_month:
                return no_update, no_update, no_update
            try:
                df_month, df_next_month = get_exec_summary_data(selected_month)
            except Exception as exc:
                return f"Error querying data: {exc}", "", True
            if df_month.empty:
                msg = f"No data found for the selected month ({selected_month})."
                return msg, "", True
            try:
                summary = generate_summary(df_month, next_month_df=df_next_month)
            except Exception as exc:
                return f"Error generating summary: {exc}", "", True
            return summary, summary, False

        @self.app.callback(
            Output("exec-summary-download", "data"),
            Output("exec-summary-save-toast", "children"),
            Output("exec-summary-save-toast", "is_open"),
            Input("exec-summary-save-btn", "n_clicks"),
            State("exec-summary-text-store", "data"),
            State("exec-summary-month-filter", "value"),
            State("exec-summary-month-filter", "options"),
            prevent_initial_call=True,
        )
        def save_exec_summary(n_clicks, summary_text, selected_month, month_options):
            if not n_clicks or not summary_text:
                return no_update, no_update, False
            month_label = selected_month or "unknown"
            for opt in (month_options or []):
                if opt.get("value") == selected_month:
                    month_label = opt.get("label", selected_month)
                    break
            content = build_save_content(summary_text, month_label)
            safe_name = (selected_month or "summary").replace(" ", "_")
            filename = f"executive_summary_{safe_name}.txt"
            return (
                dict(content=content, filename=filename),
                f"Summary saved as {filename}",
                True,
            )

        @self.app.callback(
            Output("insight-exclusion-store", "data"),
            Output("insight-exclusion-table", "data", allow_duplicate=True),
            Output("insight-exclusion-save-toast", "children"),
            Output("insight-exclusion-save-toast", "is_open"),
            Input("insight-exclusion-save", "n_clicks"),
            State("insight-exclusion-table", "data"),
            State("insight-exclusion-store", "data"),
            prevent_initial_call=True,
        )
        def save_exclusion_changes(n_clicks, current_rows, original_rows):
            if not n_clicks:
                return no_update, no_update, no_update, False

            current_rows = current_rows or []
            original_rows = original_rows or []

            original_by_insight = {
                row.get("insight_type"): row
                for row in original_rows
                if row.get("insight_type") is not None
            }
            current_by_insight = {
                row.get("insight_type"): row
                for row in current_rows
                if row.get("insight_type") is not None
            }

            updated_rows = []
            for insight_type, current_row in current_by_insight.items():
                original_row = original_by_insight.get(insight_type)
                if original_row is None:
                    continue
                orig_reason = str(original_row.get("exclusion_reason") or "")
                curr_reason = str(current_row.get("exclusion_reason") or "")
                if orig_reason != curr_reason:
                    updated_rows.append((original_row, current_row))

            if not updated_rows:
                return no_update, no_update, "No changes to save.", True

            try:
                apply_exclusion_changes(updated_rows)
            except Exception as exc:
                return no_update, no_update, f"Save failed: {exc}", True

            merged = dict(original_by_insight)
            for insight_type, row in current_by_insight.items():
                merged[insight_type] = row

            merged_list = list(merged.values())
            return merged_list, merged_list, f"Saved {len(updated_rows)} updated row(s).", True