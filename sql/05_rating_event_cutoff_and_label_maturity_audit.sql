-- Overall coverage of the issuer-rating table
SELECT
    MIN(ANNC_DT) AS min_annc_dt,
    MAX(ANNC_DT) AS max_annc_dt,
    COUNT(*) AS total_records,
    COUNT(DISTINCT INST_CODE) AS institutions
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND ANNC_DT IS NOT NULL;
  
-- Coverage of downgrade events
SELECT
    MIN(ANNC_DT) AS min_downgrade_annc_dt,
    MAX(ANNC_DT) AS max_downgrade_annc_dt,
    COUNT(*) AS downgrade_records,
    COUNT(DISTINCT INST_CODE) AS downgraded_institutions
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND RT_CHG = '4'
  AND ANNC_DT IS NOT NULL;
  
-- Check whether rating data remain after the 2025Q3 label window
SELECT
    COUNT(*) AS rating_n_after,
    COUNT(DISTINCT INST_CODE) AS inst_n_after
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND ANNC_DT > 20260331;
  
-- Check the six-month 2025Q3 label window
SELECT
    COUNT(*) AS downgrade_n,
    COUNT(DISTINCT INST_CODE) AS downgrade_inst_n,
    MIN(ANNC_DT) AS first_event_dt,
    MAX(ANNC_DT) AS last_event_dt
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND RT_CHG = '4'
  AND ANNC_DT > 20250930
  AND ANNC_DT <= 20260331;

-- Check for event records after the window ends
SELECT
    COUNT(*) AS downgrade_n_after,
    COUNT(DISTINCT INST_CODE) AS inst_n_after,
    MIN(ANNC_DT) AS first_event_after,
    MAX(ANNC_DT) AS last_event_after
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND RT_CHG = '4'
  AND ANNC_DT > 20260331;

-- Check data continuity by month
SELECT
    SUBSTR(TO_CHAR(ANNC_DT), 1, 6) AS event_month,
    COUNT(*) AS rating_records,
    SUM(
        CASE
            WHEN RT_CHG = '4'
            THEN 1
            ELSE 0
        END
    ) AS downgrade_records,
    COUNT(DISTINCT INST_CODE) AS institutions
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND ANNC_DT BETWEEN 20251001 AND 20260831
GROUP BY
    SUBSTR(TO_CHAR(ANNC_DT), 1, 6)
ORDER BY event_month;

-- Check within-window event deduplication
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
HAVING COUNT(*) > 1
ORDER BY
    ANNC_DT,
    INST_CODE;
    
-- Aggregated-only version
SELECT
    COUNT(*) AS raw_records,
    COUNT(
        DISTINCT INST_CODE
    ) AS unique_inst,
    COUNT(
        DISTINCT INST_CODE
        || '|'
        || TO_CHAR(ANNC_DT)
    ) AS unique_inst_date
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND RT_CHG = '4'
  AND ANNC_DT > 20250930
  AND ANNC_DT <= 20260331;

-- Reconcile the 2025Q3 production point-in-time master panel with the aggregate hidden labels
WITH rating_candidates AS (
    SELECT
        r.INST_CODE,
        ROW_NUMBER() OVER (
            PARTITION BY r.INST_CODE
            ORDER BY
                r.ANNC_DT DESC,
                r.RT_DT DESC,
                r.DS_ID DESC
        ) AS rn
    FROM MDB.IV_BOND_INST_RATING r
    WHERE r.DS_SRC = 'JY'
      AND r.RT_TYPE = '3'
      AND r.END_DT >= r.START_DT
      AND r.ANNC_DT <= 20250930
      AND r.START_DT <= 20250930
      AND r.END_DT >= 20250930
),
base AS (
    SELECT INST_CODE
    FROM rating_candidates
    WHERE rn = 1
),
labeled AS (
    SELECT
        b.INST_CODE,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM MDB.IV_BOND_INST_RATING e
                WHERE e.INST_CODE = b.INST_CODE
                  AND e.DS_SRC = 'JY'
                  AND e.RT_TYPE = '3'
                  AND e.RT_CHG = '4'
                  AND e.ANNC_DT > 20250930
                  AND e.ANNC_DT <= 20260331
            )
            THEN 1
            ELSE 0
        END AS downgrade_6m
    FROM base b
)
SELECT
    COUNT(*) AS sample_n,
    COUNT(DISTINCT INST_CODE) AS inst_n,
    SUM(downgrade_6m) AS positive_n,
    ROUND(
        SUM(downgrade_6m) / COUNT(*),
        12
    ) AS event_rate
FROM labeled;
