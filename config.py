"""
Configuration file for GitLab Release Report Generator
Contains database connection parameters and email recipients
"""

from datetime import date


def is_iteration_ending_this_week(iteration_end_date_str: str) -> bool:
    """Return True when the iteration's end date falls within the same
    ISO calendar week as today (the week the DAG is currently running).

    Logic
    -----
    The DAGs fire every Tuesday and Wednesday.  We consider the scripts
    to be running "in the iteration-ending week" when the
    ``iteration_end_date`` shares the **same ISO year + ISO week number**
    as today's date.

    Examples (2-week iteration ending Mon 2026-05-19)
    --------------------------------------------------
    * Script runs Tue 2026-05-12  ->  ISO week 20
      iteration_end_date = 2026-05-19  ->  ISO week 21
      -> different week  -> returns False
      -> Previous-iteration block is SKIPPED (no DB insert).

    * Script runs Tue 2026-05-19  ->  ISO week 21
      iteration_end_date = 2026-05-19  ->  ISO week 21
      -> same week  -> returns True
      -> Previous-iteration block RUNS (release week).

    * Script runs Wed 2026-05-20  ->  ISO week 21
      iteration_end_date = 2026-05-19  ->  ISO week 21
      -> same week  -> returns True  (Wed is still in the same ISO week).

    Fail-safe
    ---------
    If the date string cannot be parsed the function returns True so
    that the script does not silently skip a run due to a bad date value.

    Parameters
    ----------
    iteration_end_date_str : str
        Iteration end date in 'YYYY-MM-DD' format
        (as returned by the GitLab API due_date field).

    Returns
    -------
    bool
    """
    try:
        iteration_end = date.fromisoformat(iteration_end_date_str)
    except (ValueError, TypeError):
        # Fail-safe: allow the run so we do not silently drop data.
        return True

    today = date.today()
    # isocalendar() returns (iso_year, iso_week, iso_weekday)
    today_iso_year, today_iso_week, _ = today.isocalendar()
    end_iso_year, end_iso_week, _ = iteration_end.isocalendar()

    return (today_iso_year == end_iso_year) and (today_iso_week == end_iso_week)


# ---------------------------------------------------------------------------
# Greenplum Database Configuration
# ---------------------------------------------------------------------------
GREENPLUM_HOST = 'greenplum-rdsp.zur.swissbank.com'
GREENPLUM_PORT = 5432
GREENPLUM_DB = 'gprdsp'
GREENPLUM_USER = 'ds_rdsp_dev'

IKG_STATUS = {1: "Sent to IKG", 2: "Not sent to IKG", 3: "Waiting to send to IKG",
              4: "Backlog-Undefined", 5: "Backlog-defined", 6: "under evaluation",
              7: "under review"}
DS_LIST = ["Dorota", "Jarek", "Michal", "Kasia", "Angel", "Kuba", "Sarah", "Gulnar", "Maria"]
PO_LIST = ["Greg", "Jake", "Danny", "Soorya", "Vivian", "Krzysztof", "John", "Rafal",
           "Joanna", "Vivek", "Peter"]
