import datetime
import getpass
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import connection as PGConnection


TARGET_SCHEMA = "sandbox_prj_smart_insights"
METADATA_TABLE = "ikg_table_lineage_metadata_auto_refresh"
RULE_METADATA_TABLE = "odm_rule_metadata_auto_refresh"
OUTPUT_TABLE = "ikg_table_lineage_auto_refresh_temp"
OUTPUT_PREFIX = "ikg_table_lineage"
OUTPUT_SUFFIX = "temp"
TARGET_OWNER = "erd_gpdb_prj_smart_insights"
TARGET_READER = "erd_gpdb_prj_smart_insights_ro"
DB_BASE_CONFIG = {
    "host": "greenplum-rdsp.zur.swissbank.com",
    "port": "5432",
    "dbname": "gprdsp",
    "user": "ds_rdsp_dev",
}
LINEAGE_TEMP_COLUMNS = [
    "order_index",
    "root_profile_table",
    "target_table",
    "source_schema",
    "source_table",
    "process",
    "filename",
    "filepath",
    "level",
    "current_timestamp",
]


@dataclass
class MetadataRow:
    filename: str
    filepath: str
    process: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]


@dataclass
class DependencyRow:
    order_index: int
    root_profile_table: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]
    process: str
    filename: str
    filepath: str
    level: int
    current_timestamp: datetime.datetime


def qualified_table(table_name: str) -> sql.SQL:
    return sql.SQL("{}.{}").format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(table_name))


def fetch_profile_tables(conn: PGConnection, insight_types: Sequence[str]) -> List[str]:
    query = sql.SQL(
        "SELECT DISTINCT profile_table FROM {} WHERE insight_type = ANY(%s)"
    ).format(qualified_table(RULE_METADATA_TABLE))
    with conn.cursor() as cur:
        cur.execute(query, (insight_types,))
        rows = [row[0] for row in cur.fetchall() if row[0]]
    # preserve order while removing duplicates
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in rows:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def fetch_metadata_rows(conn: PGConnection) -> List[MetadataRow]:
    query = sql.SQL(
        """
        SELECT filename, filepath, process, target_table, source_schema, source_table
        FROM {}
        """
    ).format(qualified_table(METADATA_TABLE))
    with conn.cursor() as cur:
        cur.execute(query)
        return [
            MetadataRow(
                filename=row[0],
                filepath=row[1],
                process=row[2],
                target_table=row[3],
                source_schema=row[4],
                source_table=row[5],
            )
            for row in cur.fetchall()
        ]


def build_adjacency(metadata_rows: Sequence[MetadataRow]) -> Dict[str, List[MetadataRow]]:
    adjacency: Dict[str, List[MetadataRow]] = defaultdict(list)
    for row in metadata_rows:
        if not row.target_table:
            continue
        adjacency[row.target_table.lower()].append(row)
    return adjacency


def build_lineage_rows(
    profile_tables: Sequence[str],
    adjacency: Dict[str, List[MetadataRow]],
    run_timestamp: datetime.datetime,
) -> List[DependencyRow]:
    results: List[DependencyRow] = []
    order_index = 1

    def dfs(
        current_target: str,
        root_target: str,
        level: int,
        path: Set[str],
        edge_seen: Set[Tuple[str, str, str]],
    ) -> None:
        nonlocal order_index
        if not current_target:
            return
        current_lower = current_target.lower()
        if current_lower in path:
            logging.debug("Cycle detected at %s under root %s", current_target, root_target)
            return
        path.add(current_lower)
        rows = adjacency.get(current_lower, [])
        for row in rows:
            source = row.source_table
            if not source:
                continue
            source_lower = source.lower()
            edge_key = (root_target.lower(), row.target_table.lower(), source_lower)
            if edge_key in edge_seen:
                continue
            if source_lower in path:
                logging.debug(
                    "Cycle detected: %s -> %s for root %s", row.target_table, source, root_target
                )
                continue
            dfs(source, root_target, level + 1, path, edge_seen)
            edge_seen.add(edge_key)
            results.append(
                DependencyRow(
                    order_index=order_index,
                    root_profile_table=root_target,
                    target_table=row.target_table,
                    source_schema=row.source_schema,
                    source_table=source,
                    process=row.process,
                    filename=row.filename,
                    filepath=row.filepath,
                    level=level,
                    current_timestamp=run_timestamp,
                )
            )
            order_index += 1
        path.remove(current_lower)

    for root in profile_tables:
        edge_seen: Set[Tuple[str, str, str]] = set()
        dfs(root, root, 1, set(), edge_seen)
        template_rows = adjacency.get(root.lower(), [])
        template = template_rows[0] if template_rows else None
        results.append(
            DependencyRow(
                order_index=order_index,
                root_profile_table=root,
                target_table=root,
                source_schema=None,
                source_table=None,
                process=template.process if template else "",
                filename=template.filename if template else "",
                filepath=template.filepath if template else "",
                level=0,
                current_timestamp=run_timestamp,
            )
        )
        order_index += 1

    return results


def rows_to_dataframe(rows: Sequence[DependencyRow]) -> pd.DataFrame:
    records = [
        {
            "order_index": row.order_index,
            "root_profile_table": row.root_profile_table,
            "target_table": row.target_table,
            "source_schema": row.source_schema,
            "source_table": row.source_table,
            "process": row.process,
            "filename": row.filename,
            "filepath": row.filepath,
            "level": row.level,
            "current_timestamp": row.current_timestamp,
        }
        for row in rows
    ]
    return pd.DataFrame(records, columns=LINEAGE_TEMP_COLUMNS)


def write_temp_excel(df: pd.DataFrame, run_timestamp: datetime.datetime) -> str:
    timestamp_str = run_timestamp.strftime("%Y%m%d%H%M%S")
    output_file = f"{OUTPUT_PREFIX}_{timestamp_str}_{OUTPUT_SUFFIX}.xlsx"
    df.to_excel(output_file, index=False)
    logging.info("Wrote %s", output_file)
    return output_file


def refresh_temp_table(conn: PGConnection, rows: Sequence[DependencyRow]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_table(OUTPUT_TABLE))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    order_index INTEGER,
                    root_profile_table TEXT,
                    target_table TEXT,
                    source_schema TEXT,
                    source_table TEXT,
                    process TEXT,
                    filename TEXT,
                    filepath TEXT,
                    level INTEGER,
                    "current_timestamp" TIMESTAMP
                )
                """
            ).format(qualified_table(OUTPUT_TABLE))
        )
        values = [
            (
                row.order_index,
                row.root_profile_table,
                row.target_table,
                row.source_schema,
                row.source_table,
                row.process,
                row.filename,
                row.filepath,
                row.level,
                row.current_timestamp,
            )
            for row in rows
        ]
        if values:
            execute_values(
                cur,
                sql.SQL(
                    'INSERT INTO {} (order_index, root_profile_table, target_table, source_schema, source_table, process, filename, filepath, level, "current_timestamp") VALUES %s'
                ).format(qualified_table(OUTPUT_TABLE)),
                values,
            )
        cur.execute(
            sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                qualified_table(OUTPUT_TABLE), sql.Identifier(TARGET_OWNER)
            )
        )
        cur.execute(
            sql.SQL("GRANT SELECT ON {} TO {}").format(
                qualified_table(OUTPUT_TABLE), sql.Identifier(TARGET_READER)
            )
        )
    conn.commit()


def build_db_config(password: str) -> Dict[str, str]:
    config = DB_BASE_CONFIG.copy()
    config["password"] = password
    return config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    insight_raw = input("Enter insight_type: ").strip()
    insight_types = [part.strip() for part in insight_raw.split(",") if part.strip()]
    if not insight_types:
        raise ValueError("At least one insight_type is required.")

    db_password = getpass.getpass("Enter Password for DB User: ")
    db_config = build_db_config(db_password)

    with psycopg2.connect(**db_config) as conn:
        profile_tables = fetch_profile_tables(conn, insight_types)
        if not profile_tables:
            logging.warning("No profile tables found for provided insight types.")
            return
        metadata_rows = fetch_metadata_rows(conn)

    adjacency = build_adjacency(metadata_rows)
    run_timestamp = datetime.datetime.utcnow()
    lineage_rows = build_lineage_rows(profile_tables, adjacency, run_timestamp)

    df = rows_to_dataframe(lineage_rows)
    write_temp_excel(df, run_timestamp)

    with psycopg2.connect(**db_config) as conn:
        refresh_temp_table(conn, lineage_rows)


if __name__ == "__main__":
    main()
