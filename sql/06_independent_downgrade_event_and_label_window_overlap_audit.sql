-- 06_independent_downgrade_event_and_label_window_overlap_audit.sql
--
-- Audit status:
-- The Oracle event-union query across all 11 quarters was stopped because of excessive client runtime.
-- It produced no result and was not used as report evidence.
--
-- A post hoc aggregate audit of overlap across all TEST positive snapshots used the de-identified modeling export:
-- 393 positive issuer-quarter snapshots involve 171 unique anonymized issuers.
--
-- The Oracle event-level audit completed successfully only for 2025Q3:
-- 11,418 prediction units, 10 positive snapshots, 10 institutions,
-- 12 unique issuer-announcement-date events, and 12 raw records.
--
-- This file retains only successfully executed read-only queries used as report evidence.
-- It does not output issuer identifiers.

-- Single-quarter split test: 2025Q3
-- Return aggregated results only; do not output issuer identifiers.

WITH base AS (
    SELECT DISTINCT
        INST_CODE
    FROM MDB.IV_BOND_INST_RATING
    WHERE DS_SRC = 'JY'
      AND RT_TYPE = '3'
      AND END_DT >= START_DT
      AND ANNC_DT <= 20250930
      AND START_DT <= 20250930
      AND END_DT >= 20250930
),
event_group AS (
    SELECT
        INST_CODE,
        ANNC_DT,
        COUNT(*) AS raw_n
    FROM MDB.IV_BOND_INST_RATING
    WHERE DS_SRC = 'JY'
      AND RT_TYPE = '3'
      AND RT_CHG = '4'
      AND ANNC_DT > 20250930
      AND ANNC_DT <= 20260331
    GROUP BY
        INST_CODE,
        ANNC_DT
),
links AS (
    SELECT
        b.INST_CODE,
        e.ANNC_DT,
        e.raw_n
    FROM base b
    JOIN event_group e
      ON e.INST_CODE = b.INST_CODE
),
positive_inst AS (
    SELECT DISTINCT
        INST_CODE
    FROM links
)
SELECT
    (
        SELECT COUNT(*)
        FROM base
    ) AS base_n,

    (
        SELECT COUNT(*)
        FROM positive_inst
    ) AS positive_snapshot_n,

    (
        SELECT COUNT(*)
        FROM positive_inst
    ) AS positive_inst_n,

    (
        SELECT COUNT(*)
        FROM links
    ) AS issuer_date_event_n,

    (
        SELECT SUM(raw_n)
        FROM links
    ) AS raw_record_n
FROM dual;
