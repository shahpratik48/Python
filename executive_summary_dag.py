# dags/ikg/sprint_report/executive_summary_dag.py
# -*- coding: utf-8 -*-
"""Monthly Executive Summary DAG.

Schedule : 4th Monday of every month at 08:00 AM EST.
           Cron  0 8 22-28 * 1
           The 4th Monday of any month always falls on the 22nd–28th, so
           combining day-of-week=Monday with day-of-month=22-28 fires
           exactly once per month.

Task flow: start → generate_executive_summary → end

The PythonOperator calls generate_executive_summary() from
scripts/executive_summary.py.  That function:
  1. Derives the current year/month from datetime.now().
  2. Queries Greenplum for release data, reactivated insights, and
     next-month look-ahead data.
  3. Calls the OpenAI API to generate a structured executive summary.
  4. Saves the output to  scripts/output/executive_summary_YYYY-MM.txt.
"""

from pathlib import Path
import sys
import importlib

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

try:
    from airflow.operators.empty import EmptyOperator
except Exception:
    from airflow.operators.dummy import DummyOperator as EmptyOperator


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DAG_ID             = "executive_summary_monthly"
SCRIPT_MODULE_NAME = "executive_summary"

default_args = {
    "owner"           : "STAAT",
    "email_on_failure": True,
    "email_on_retry"  : False,
    "TAG"             : "IKG",
}

# -------------------------------------------------------------------
# Import script from ./scripts
# -------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))

_exec_summary_mod          = importlib.import_module(SCRIPT_MODULE_NAME)
generate_executive_summary = getattr(_exec_summary_mod, "generate_executive_summary")

# -------------------------------------------------------------------
# DAG – 4th Monday of every month at 08:00 AM EST
#
# Cron:  0 8 22-28 * 1
#   minute      : 0
#   hour        : 8   (08:00)
#   day-of-month: 22-28  (4th Monday always lands here)
#   month       : *
#   day-of-week : 1   (Monday)
# -------------------------------------------------------------------
with DAG(
    dag_id            =DAG_ID,
    description       ="Monthly Executive Summary – 4th Monday 8 AM EST",
    schedule_interval ="0 8 22-28 * 1",
    start_date        =pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup           =False,
    default_args      =default_args,
    tags              =["IKG"],
    render_template_as_native_obj=True,
    max_active_runs   =1,
) as dag:

    start = EmptyOperator(task_id="start")

    run_exec_summary = PythonOperator(
        task_id        ="generate_executive_summary",
        python_callable=generate_executive_summary,
    )

    end = EmptyOperator(task_id="end")

    start >> run_exec_summary >> end
