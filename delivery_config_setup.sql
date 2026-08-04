-- ============================================================================
-- delivery_config_setup.sql
--
-- ONE-TIME setup for the pod & crew delivery productivity reporting.
-- Creates the two configuration tables and loads the current pods.
--
--   sandbox_prj_smart_insights.delivery_org_hierarchy   one row per POD, with its full org path
--   sandbox_prj_smart_insights.delivery_pod_links       one row per LINK, so a pod may have many
--
-- Two tables rather than one because the relationship differs: a pod has
-- exactly one place in the hierarchy but many links. Flattening both into a
-- single table would repeat the whole org path on every URL.
--
-- Safe to re-run: the inserts are preceded by DELETEs, so the configuration
-- is replaced wholesale. A pod removed from this script really does
-- disappear rather than lingering. The metric tables are never touched.
--
-- Run as:  psql -h greenplum-rdsp.zur.swissbank.com -p 5432 -d gprdsp \
--               -U ds_rdsp_dev -f delivery_config_setup.sql
-- ============================================================================

\set ON_ERROR_STOP on

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Tables
--    The schema already exists and is managed elsewhere, so it is not created.
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sandbox_prj_smart_insights.delivery_org_hierarchy (
  division           varchar(200),   -- e.g. Global Wealth Management Americas
  subdivision        varchar(200),
  stream             varchar(200),
  crew               varchar(200),   -- the crew roll-up is produced per crew
  pod_name           varchar(200),   -- display name; also the report file prefix
  pod_key            varchar(100),   -- joins to delivery_pod_links; must be unique
  group_path         varchar(500),   -- group holding iterations/epics
  is_active          boolean DEFAULT true,   -- false retires a pod, keeping history
  updated_timestamp  timestamp
)
DISTRIBUTED BY (pod_key);

CREATE TABLE IF NOT EXISTS sandbox_prj_smart_insights.delivery_pod_links (
  pod_key            varchar(100),   -- joins to delivery_org_hierarchy
  link_type          varchar(10),    -- 'issue' or 'mr'
  link_label         varchar(200),   -- MR column header; ignored for issue links
  link_path          varchar(500),   -- project path or group path, no https:// prefix
  is_active          boolean DEFAULT true,
  updated_timestamp  timestamp
)
DISTRIBUTED BY (pod_key);

-- ----------------------------------------------------------------------------
-- 2. Ownership and read access
-- ----------------------------------------------------------------------------

ALTER TABLE sandbox_prj_smart_insights.delivery_org_hierarchy OWNER TO erd_gpdb_prj_smart_insights;
GRANT SELECT ON sandbox_prj_smart_insights.delivery_org_hierarchy TO erd_gpdb_prj_smart_insights_ro;
ALTER TABLE sandbox_prj_smart_insights.delivery_pod_links OWNER TO erd_gpdb_prj_smart_insights;
GRANT SELECT ON sandbox_prj_smart_insights.delivery_pod_links TO erd_gpdb_prj_smart_insights_ro;

-- ----------------------------------------------------------------------------
-- 3. Configuration data
--    Replaced wholesale so re-running cannot duplicate rows.
-- ----------------------------------------------------------------------------

DELETE FROM sandbox_prj_smart_insights.delivery_pod_links;
DELETE FROM sandbox_prj_smart_insights.delivery_org_hierarchy;

INSERT INTO sandbox_prj_smart_insights.delivery_org_hierarchy
  (division, subdivision, stream, crew, pod_name, pod_key, group_path,
   is_active, updated_timestamp)
VALUES
  ('Global Wealth Management Americas', 'Global Wealth Management Americas',
   'Data Analytics and Foundational Platforms',
   'STAAT Data Science', 'Insights', 'insights',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons', true, now()),
  ('Global Wealth Management Americas', 'Global Wealth Management Americas',
   'Data Analytics and Foundational Platforms',
   'STAAT Data Science', 'Conv. Insights', 'conv_insights',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform', true, now()),
  ('Global Wealth Management Americas', 'Global Wealth Management Americas',
   'Data Analytics and Foundational Platforms',
   'STAAT Data Science', 'DS Data', 'ds_data',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-data/ds-data', true, now());

INSERT INTO sandbox_prj_smart_insights.delivery_pod_links
  (pod_key, link_type, link_label, link_path, is_active, updated_timestamp)
VALUES
  -- insights
  ('insights', 'issue', '',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons/staat-ds-insights-home', true, now()),
  ('insights', 'mr', 'IKG',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ikg-dags', true, now()),
  ('insights', 'mr', 'NLG',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags', true, now()),
  ('insights', 'mr', 'ODM',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/odm-dags', true, now()),
  -- conv_insights
  ('conv_insights', 'issue', '',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/data-elements', true, now()),
  ('conv_insights', 'mr', 'conv-insights',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/conv-insights', true, now()),
  ('conv_insights', 'mr', 'data-elements',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/data-elements', true, now()),
  ('conv_insights', 'mr', 'data-elements-common',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/data-elements-common', true, now()),
  ('conv_insights', 'mr', 'de-api',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/de-api', true, now()),
  ('conv_insights', 'mr', 'transcript-accumulator',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/transcript-accumulator', true, now()),
  ('conv_insights', 'mr', 'nlg-chat',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-chat', true, now()),
  ('conv_insights', 'mr', 'portfolioq',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/portfolioq', true, now()),
  -- ds_data
  ('ds_data', 'issue', '',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-data/ds-data/staat-ds-data-engineering', true, now()),
  ('ds_data', 'mr', 'ds-data-dags',
   'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ds-data-dags', true, now());

COMMIT;

-- ----------------------------------------------------------------------------
-- 4. Verify
-- ----------------------------------------------------------------------------

SELECT h.division, h.stream, h.crew, h.pod_name, h.pod_key,
       SUM(CASE WHEN l.link_type = 'issue' THEN 1 ELSE 0 END) AS issue_links,
       SUM(CASE WHEN l.link_type = 'mr'    THEN 1 ELSE 0 END) AS mr_links
FROM        sandbox_prj_smart_insights.delivery_org_hierarchy h
LEFT JOIN   sandbox_prj_smart_insights.delivery_pod_links     l
       ON   l.pod_key = h.pod_key AND l.is_active
WHERE  h.is_active
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 2, 3, 4;

-- Expected: 3 pods --  Insights 1/3,  Conv. Insights 1/7,  DS Data 1/1
-- A pod needs at least one ACTIVE issue link and one ACTIVE mr link, or the
-- reporting job logs it and skips it rather than producing an empty report.
