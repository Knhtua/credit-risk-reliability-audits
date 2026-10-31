-- Construct the quarterly sample skeleton and count observations and positive cases per quarter
WITH quarters AS (
    SELECT
        ADD_MONTHS(
            DATE '2013-03-31',
            3 * (LEVEL - 1)
        ) AS snapshot_dt
    FROM dual
    CONNECT BY ADD_MONTHS(
        DATE '2013-03-31',
        3 * (LEVEL - 1)
    ) <= DATE '2025-09-30'
),
rated_universe AS (
    SELECT
        q.snapshot_dt,
        r.INST_CODE,
        MAX(r.ANNC_DT) AS latest_rating_dt
    FROM quarters q
    JOIN MDB.IV_BOND_INST_RATING r
      ON r.DS_SRC = 'JY'
     AND r.RT_TYPE = '3'
     AND r.ANNC_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
    GROUP BY
        q.snapshot_dt,
        r.INST_CODE
),
labeled AS (
    SELECT
        u.snapshot_dt,
        u.INST_CODE,
        u.latest_rating_dt,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM MDB.IV_BOND_INST_RATING e
                WHERE e.INST_CODE = u.INST_CODE
                  AND e.DS_SRC = 'JY'
                  AND e.RT_TYPE = '3'
                  AND e.RT_CHG = '4'
                  AND e.ANNC_DT >
                      TO_NUMBER(
                          TO_CHAR(u.snapshot_dt, 'YYYYMMDD')
                      )
                  AND e.ANNC_DT <=
                      TO_NUMBER(
                          TO_CHAR(
                              ADD_MONTHS(u.snapshot_dt, 6),
                              'YYYYMMDD'
                          )
                      )
            )
            THEN 1
            ELSE 0
        END AS downgrade_6m
    FROM rated_universe u
)
SELECT
    TO_CHAR(snapshot_dt, 'YYYY-MM-DD') AS snapshot_dt,
    COUNT(*) AS sample_count,
    SUM(downgrade_6m) AS positive_count,
    ROUND(
        100 * SUM(downgrade_6m) / COUNT(*),
        4
    ) AS positive_pct
FROM labeled
GROUP BY snapshot_dt
ORDER BY snapshot_dt;

-- Audit the timeliness of active ratings by inspecting rating-validity fields
SELECT
    COUNT(*) AS total_records,

    SUM(
        CASE WHEN START_DT IS NULL
        THEN 1 ELSE 0 END
    ) AS missing_start_dt,

    SUM(
        CASE WHEN END_DT IS NULL
        THEN 1 ELSE 0 END
    ) AS missing_end_dt,

    SUM(
        CASE
            WHEN START_DT IS NOT NULL
             AND END_DT IS NOT NULL
             AND END_DT < START_DT
            THEN 1 ELSE 0
        END
    ) AS end_before_start,

    MIN(START_DT) AS min_start_dt,
    MAX(START_DT) AS max_start_dt,

    MIN(END_DT) AS min_end_dt,
    MAX(END_DT) AS max_end_dt,

    SUM(
        CASE
            WHEN START_DT IS NOT NULL
             AND START_DT < 19900101
            THEN 1 ELSE 0
        END
    ) AS invalid_start_dt,

    SUM(
        CASE
            WHEN END_DT IS NOT NULL
             AND END_DT < 19900101
            THEN 1 ELSE 0
        END
    ) AS invalid_end_dt

FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3';
  
-- Rebuild the at-risk issuer set by rating validity and recompute quarterly samples and positive rates
WITH quarters AS (
    SELECT
        ADD_MONTHS(
            DATE '2013-03-31',
            3 * (LEVEL - 1)
        ) AS snapshot_dt
    FROM dual
    CONNECT BY ADD_MONTHS(
        DATE '2013-03-31',
        3 * (LEVEL - 1)
    ) <= DATE '2025-09-30'
),
active_universe AS (
    SELECT DISTINCT
        q.snapshot_dt,
        r.INST_CODE
    FROM quarters q
    JOIN MDB.IV_BOND_INST_RATING r
      ON r.DS_SRC = 'JY'
     AND r.RT_TYPE = '3'
     AND r.END_DT >= r.START_DT
     AND r.ANNC_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.START_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.END_DT >= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
),
labeled AS (
    SELECT
        u.snapshot_dt,
        u.INST_CODE,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM MDB.IV_BOND_INST_RATING e
                WHERE e.INST_CODE = u.INST_CODE
                  AND e.DS_SRC = 'JY'
                  AND e.RT_TYPE = '3'
                  AND e.RT_CHG = '4'
                  AND e.ANNC_DT >
                      TO_NUMBER(
                          TO_CHAR(u.snapshot_dt, 'YYYYMMDD')
                      )
                  AND e.ANNC_DT <=
                      TO_NUMBER(
                          TO_CHAR(
                              ADD_MONTHS(u.snapshot_dt, 6),
                              'YYYYMMDD'
                          )
                      )
            )
            THEN 1
            ELSE 0
        END AS downgrade_6m
    FROM active_universe u
)
SELECT
    TO_CHAR(snapshot_dt, 'YYYY-MM-DD') AS snapshot_dt,
    COUNT(*) AS active_inst,
    SUM(downgrade_6m) AS positive_count,
    ROUND(
        100 * SUM(downgrade_6m) / COUNT(*),
        4
    ) AS positive_pct
FROM labeled
GROUP BY snapshot_dt
ORDER BY snapshot_dt;

-- Build the issuer-quarter master panel for ratings and labels
-- For each issuer and quarter-end, select the latest rating then public and valid, then create a six-month-forward label
WITH quarters AS (
    SELECT
        ADD_MONTHS(
            DATE '2013-03-31',
            3 * (LEVEL - 1)
        ) AS snapshot_dt
    FROM dual
    CONNECT BY ADD_MONTHS(
        DATE '2013-03-31',
        3 * (LEVEL - 1)
    ) <= DATE '2025-09-30'
),
rating_candidates AS (
    SELECT
        q.snapshot_dt,
        r.INST_CODE,
        r.INST_NAME,
        r.INST_TYPE,
        r.RT_DT,
        r.ANNC_DT,
        r.START_DT,
        r.END_DT,
        r.RT_RANK,
        r.PREV_RANK,
        r.RT_CHG,
        r.RT_EXP,
        r.RT_INST_CODE,
        r.RT_INST_NAME,
        r.DS_ID,

        ROW_NUMBER() OVER (
            PARTITION BY
                q.snapshot_dt,
                r.INST_CODE
            ORDER BY
                r.ANNC_DT DESC,
                r.RT_DT DESC,
                r.DS_ID DESC
        ) AS rn

    FROM quarters q
    JOIN MDB.IV_BOND_INST_RATING r
      ON r.DS_SRC = 'JY'
     AND r.RT_TYPE = '3'
     AND r.END_DT >= r.START_DT
     AND r.ANNC_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.START_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.END_DT >= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
),
base AS (
    SELECT *
    FROM rating_candidates
    WHERE rn = 1
)
SELECT
    TO_CHAR(b.snapshot_dt, 'YYYY-MM-DD') AS snapshot_dt,
    b.INST_CODE,
    b.INST_NAME,
    b.INST_TYPE,

    b.RT_RANK AS current_rank,
    b.PREV_RANK AS previous_rank,
    b.RT_CHG AS latest_rt_chg,
    b.RT_EXP AS rating_outlook,

    b.RT_INST_CODE AS rating_agency_code,
    b.RT_INST_NAME AS rating_agency_name,

    b.RT_DT AS latest_rating_dt,
    b.ANNC_DT AS latest_annc_dt,
    b.START_DT AS rating_start_dt,
    b.END_DT AS rating_end_dt,

    TRUNC(
        b.snapshot_dt
        - TO_DATE(
            TO_CHAR(b.ANNC_DT),
            'YYYYMMDD'
        )
    ) AS rating_age_days,

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM MDB.IV_BOND_INST_RATING e
            WHERE e.INST_CODE = b.INST_CODE
              AND e.DS_SRC = 'JY'
              AND e.RT_TYPE = '3'
              AND e.RT_CHG = '4'
              AND e.ANNC_DT >
                  TO_NUMBER(
                      TO_CHAR(
                          b.snapshot_dt,
                          'YYYYMMDD'
                      )
                  )
              AND e.ANNC_DT <=
                  TO_NUMBER(
                      TO_CHAR(
                          ADD_MONTHS(
                              b.snapshot_dt,
                              6
                          ),
                          'YYYYMMDD'
                      )
                  )
        )
        THEN 1
        ELSE 0
    END AS downgrade_6m

FROM base b
ORDER BY
    b.snapshot_dt,
    b.INST_CODE;
    
-- Audit financial-statement definitions
-- Assess overall financial-table quality
SELECT
    'BALANCE' AS table_src,
    COUNT(*) AS total_records,
    COUNT(DISTINCT INST_ID) AS institutions,
    SUM(
        CASE WHEN ACTL_ANNC_DT IS NULL
        THEN 1 ELSE 0 END
    ) AS missing_actl_annc,
    SUM(
        CASE WHEN RPT_DT IS NULL
        THEN 1 ELSE 0 END
    ) AS missing_rpt_dt,
    MIN(RPT_DT) AS min_rpt_dt,
    MAX(RPT_DT) AS max_rpt_dt,
    MIN(ACTL_ANNC_DT) AS min_actl_annc,
    MAX(ACTL_ANNC_DT) AS max_actl_annc
FROM MDB.IP_INST_FINC_BALANCE

UNION ALL

SELECT
    'INCOME',
    COUNT(*),
    COUNT(DISTINCT INST_ID),
    SUM(
        CASE WHEN ACTL_ANNC_DT IS NULL
        THEN 1 ELSE 0 END
    ),
    SUM(
        CASE WHEN RPT_DT IS NULL
        THEN 1 ELSE 0 END
    ),
    MIN(RPT_DT),
    MAX(RPT_DT),
    MIN(ACTL_ANNC_DT),
    MAX(ACTL_ANNC_DT)
FROM MDB.IP_INST_FINC_INCOME

UNION ALL

SELECT
    'CASHFLOW',
    COUNT(*),
    COUNT(DISTINCT INST_ID),
    SUM(
        CASE WHEN ACTL_ANNC_DT IS NULL
        THEN 1 ELSE 0 END
    ),
    SUM(
        CASE WHEN RPT_DT IS NULL
        THEN 1 ELSE 0 END
    ),
    MIN(RPT_DT),
    MAX(RPT_DT),
    MIN(ACTL_ANNC_DT),
    MAX(ACTL_ANNC_DT)
FROM MDB.IP_INST_FINC_CASHFLOW;
-- Check statement types, currencies, and data sources
SELECT
    'BALANCE' AS table_src,
    DS_SRC,
    RPT_TYPE,
    CURR_CODE,
    COUNT(*) AS record_count,
    COUNT(DISTINCT INST_ID) AS institution_count,
    MIN(RPT_DT) AS min_rpt_dt,
    MAX(RPT_DT) AS max_rpt_dt
FROM MDB.IP_INST_FINC_BALANCE
GROUP BY
    DS_SRC,
    RPT_TYPE,
    CURR_CODE

UNION ALL

SELECT
    'INCOME' AS table_src,
    DS_SRC,
    RPT_TYPE,
    CURR_CODE,
    COUNT(*) AS record_count,
    COUNT(DISTINCT INST_ID) AS institution_count,
    MIN(RPT_DT) AS min_rpt_dt,
    MAX(RPT_DT) AS max_rpt_dt
FROM MDB.IP_INST_FINC_INCOME
GROUP BY
    DS_SRC,
    RPT_TYPE,
    CURR_CODE

UNION ALL

SELECT
    'CASHFLOW' AS table_src,
    DS_SRC,
    RPT_TYPE,
    CURR_CODE,
    COUNT(*) AS record_count,
    COUNT(DISTINCT INST_ID) AS institution_count,
    MIN(RPT_DT) AS min_rpt_dt,
    MAX(RPT_DT) AS max_rpt_dt
FROM MDB.IP_INST_FINC_CASHFLOW
GROUP BY
    DS_SRC,
    RPT_TYPE,
    CURR_CODE

ORDER BY
    1,
    5 DESC;
-- Identify the RPT_TYPE dictionary entry
SELECT
    TABLE_NAME,
    COLUMN_NAME,
    COMMENTS
FROM ALL_COL_COMMENTS
WHERE OWNER = 'MDB'
  AND TABLE_NAME IN (
      'IP_INST_FINC_BALANCE',
      'IP_INST_FINC_INCOME',
      'IP_INST_FINC_CASHFLOW'
  )
  AND COLUMN_NAME = 'RPT_TYPE'
ORDER BY TABLE_NAME;
-- Query the complete dictionary directly with PRNT_ID='RPT_TYPE'
SELECT
    DS_ID,
    PRNT_ID,
    CODE,
    NAME,
    MEMO,
    DICT_TYPE,
    STATUS
FROM MDB.BIZ_DICTIONARY
WHERE PRNT_ID = 'RPT_TYPE'
ORDER BY
    CASE
        WHEN REGEXP_LIKE(CODE, '^[0-9]+$')
        THEN TO_NUMBER(CODE)
    END,
    CODE;
-- PRNT_ID='RPT_TYPE' returns zero rows in BIZ_DICTIONARY
-- Query ETL_DICT_MAPPING instead
SELECT
    DICT_CODE,
    DICT_NAME,
    DICT_ITEM_CODE,
    ITEM_MEMO,
    SRC_ITEM_CODE,
    SRC_ITEM_NAME,
    SRC_SYS,
    STATUS
FROM MDB.ETL_DICT_MAPPING
WHERE UPPER(DICT_CODE) = 'RPT_TYPE'
   OR DICT_NAME LIKE UNISTR('%\\62A5\\8868\\7C7B\\578B%')
ORDER BY
    SRC_SYS,
    SRC_ITEM_CODE;
    
-- Determine financial-statement definitions and audit quarterly coverage across the three financial tables
-- This aggregate query exports no financial detail; it only checks how many rated issuers have usable RPT_TYPE='11' financial data in each quarter
WITH quarters AS (
    SELECT
        ADD_MONTHS(
            DATE '2013-03-31',
            3 * (LEVEL - 1)
        ) AS snapshot_dt
    FROM dual
    CONNECT BY ADD_MONTHS(
        DATE '2013-03-31',
        3 * (LEVEL - 1)
    ) <= DATE '2025-09-30'
),
active_universe AS (
    SELECT DISTINCT
        q.snapshot_dt,
        r.INST_CODE
    FROM quarters q
    JOIN MDB.IV_BOND_INST_RATING r
      ON r.DS_SRC = 'JY'
     AND r.RT_TYPE = '3'
     AND r.END_DT >= r.START_DT
     AND r.ANNC_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.START_DT <= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
     AND r.END_DT >= TO_NUMBER(
         TO_CHAR(q.snapshot_dt, 'YYYYMMDD')
     )
),
balance_first AS (
    SELECT
        INST_ID,
        MIN(
            GREATEST(RPT_DT, ACTL_ANNC_DT)
        ) AS first_available_dt
    FROM MDB.IP_INST_FINC_BALANCE
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID
),
income_first AS (
    SELECT
        INST_ID,
        MIN(
            GREATEST(RPT_DT, ACTL_ANNC_DT)
        ) AS first_available_dt
    FROM MDB.IP_INST_FINC_INCOME
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID
),
cashflow_first AS (
    SELECT
        INST_ID,
        MIN(
            GREATEST(RPT_DT, ACTL_ANNC_DT)
        ) AS first_available_dt
    FROM MDB.IP_INST_FINC_CASHFLOW
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID
)
SELECT
    TO_CHAR(u.snapshot_dt, 'YYYY-MM-DD') AS snapshot_dt,
    COUNT(*) AS active_inst,

    SUM(
        CASE
            WHEN b.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
            THEN 1 ELSE 0
        END
    ) AS has_balance,

    SUM(
        CASE
            WHEN i.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
            THEN 1 ELSE 0
        END
    ) AS has_income,

    SUM(
        CASE
            WHEN c.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
            THEN 1 ELSE 0
        END
    ) AS has_cashflow,

    SUM(
        CASE
            WHEN b.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
             AND i.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
             AND c.first_available_dt <=
                 TO_NUMBER(TO_CHAR(u.snapshot_dt, 'YYYYMMDD'))
            THEN 1 ELSE 0
        END
    ) AS has_all_three

FROM active_universe u
LEFT JOIN balance_first b
  ON u.INST_CODE = b.INST_ID
LEFT JOIN income_first i
  ON u.INST_CODE = i.INST_ID
LEFT JOIN cashflow_first c
  ON u.INST_CODE = c.INST_ID
GROUP BY u.snapshot_dt
ORDER BY u.snapshot_dt;

-- Audit financial-statement versions for multiple records from the same reporting period
SELECT
    'BALANCE' AS table_src,
    COUNT(*) AS duplicate_groups,
    SUM(row_count - 1) AS extra_rows,
    SUM(
        CASE WHEN annc_versions > 1
        THEN 1 ELSE 0 END
    ) AS revised_groups,
    MAX(row_count) AS max_versions
FROM (
    SELECT
        INST_ID,
        RPT_DT,
        COUNT(*) AS row_count,
        COUNT(DISTINCT ACTL_ANNC_DT) AS annc_versions
    FROM MDB.IP_INST_FINC_BALANCE
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID, RPT_DT
    HAVING COUNT(*) > 1
)

UNION ALL

SELECT
    'INCOME',
    COUNT(*),
    SUM(row_count - 1),
    SUM(
        CASE WHEN annc_versions > 1
        THEN 1 ELSE 0 END
    ),
    MAX(row_count)
FROM (
    SELECT
        INST_ID,
        RPT_DT,
        COUNT(*) AS row_count,
        COUNT(DISTINCT ACTL_ANNC_DT) AS annc_versions
    FROM MDB.IP_INST_FINC_INCOME
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID, RPT_DT
    HAVING COUNT(*) > 1
)

UNION ALL

SELECT
    'CASHFLOW',
    COUNT(*),
    SUM(row_count - 1),
    SUM(
        CASE WHEN annc_versions > 1
        THEN 1 ELSE 0 END
    ),
    MAX(row_count)
FROM (
    SELECT
        INST_ID,
        RPT_DT,
        COUNT(*) AS row_count,
        COUNT(DISTINCT ACTL_ANNC_DT) AS annc_versions
    FROM MDB.IP_INST_FINC_CASHFLOW
    WHERE DS_SRC = 'JY'
      AND RPT_TYPE = '11'
    GROUP BY INST_ID, RPT_DT
    HAVING COUNT(*) > 1
);

-- Extract core financial fields into reduced raw tables rather than exporting all hundreds of fields; complete point-in-time matching, ratio construction, and missingness checks locally
-- Balance sheet
SELECT
    INST_ID,
    RPT_DT,
    ACTL_ANNC_DT,
    INST_TYPE,

    ASSETS_TOT,
    LA_TOT,
    EQY_TOT_INCL_MI,

    CUR_ASSETS_TOT,
    CUR_LA_TOT,

    BANK_CASH,
    INVENTORY,
    RCV_ACCT,

    SHRT_TERM_BORR,
    LONG_TERM_BORR,
    PY_BONDS,

    DS_ID
FROM MDB.IP_INST_FINC_BALANCE
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND RPT_DT <= 20250930
  AND ACTL_ANNC_DT <= 20250930
ORDER BY INST_ID, RPT_DT;
-- Income statement
SELECT
    INST_ID,
    RPT_DT,
    ACTL_ANNC_DT,
    INST_TYPE,

    OP_INCM_TOT,
    OP_INCM,
    OP_PROFIT,
    PROFIT_TOT,
    NET_PROFIT_INCL_MIN_INT_INC,

    EBIT,
    EBITDA,
    LESS_INT_EXP,
    LESS_FIN_EXP,

    DS_ID
FROM MDB.IP_INST_FINC_INCOME
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND RPT_DT <= 20250930
  AND ACTL_ANNC_DT <= 20250930
ORDER BY INST_ID, RPT_DT;
-- Cash-flow statement
SELECT
    INST_ID,
    RPT_DT,
    ACTL_ANNC_DT,

    NET_CF_OP,
    NET_CF_INVEST,
    NET_CF_FNC,

    CASH_EQU_END_PRD,
    CASH_PY_REPAY_DEBT,
    FREE_CASH_FLOW,

    DS_ID
FROM MDB.IP_INST_FINC_CASHFLOW
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND RPT_DT <= 20250930
  AND ACTL_ANNC_DT <= 20250930
ORDER BY INST_ID, RPT_DT;

-- Validate core financial fields and identify alternative definitions; where fields are entirely empty, check coverage of synonymous alternatives before re-exporting all detail
SELECT
    'BALANCE' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(ASSETS_TOT) AS assets_n,
    COUNT(LA_TOT) AS liabilities_n,
    COUNT(EQY_TOT_INCL_MI) AS equity_incl_mi_n,
    COUNT(EQY_TOT_EXCL_MI) AS equity_excl_mi_n,
    COUNT(LA_EQY_TOT) AS liabilities_equity_n,
    CAST(NULL AS NUMBER) AS net_profit_incl_n,
    CAST(NULL AS NUMBER) AS net_profit_excl_n,
    CAST(NULL AS NUMBER) AS total_balance_profit_n,
    CAST(NULL AS NUMBER) AS ebit_n,
    CAST(NULL AS NUMBER) AS ebitda_n,
    CAST(NULL AS NUMBER) AS operating_cashflow_n,
    CAST(NULL AS NUMBER) AS free_cashflow_n
FROM MDB.IP_INST_FINC_BALANCE
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND ACTL_ANNC_DT <= 20250930

UNION ALL

SELECT
    'INCOME' AS table_name,
    COUNT(*) AS total_rows,
    CAST(NULL AS NUMBER) AS assets_n,
    CAST(NULL AS NUMBER) AS liabilities_n,
    CAST(NULL AS NUMBER) AS equity_incl_mi_n,
    CAST(NULL AS NUMBER) AS equity_excl_mi_n,
    CAST(NULL AS NUMBER) AS liabilities_equity_n,
    COUNT(NET_PROFIT_INCL_MIN_INT_INC) AS net_profit_incl_n,
    COUNT(NET_PROFIT_EXCL_MIN_INT_INC) AS net_profit_excl_n,
    COUNT(TOT_BAL_NET_PROFIT) AS total_balance_profit_n,
    COUNT(EBIT) AS ebit_n,
    COUNT(EBITDA) AS ebitda_n,
    CAST(NULL AS NUMBER) AS operating_cashflow_n,
    CAST(NULL AS NUMBER) AS free_cashflow_n
FROM MDB.IP_INST_FINC_INCOME
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND ACTL_ANNC_DT <= 20250930

UNION ALL

SELECT
    'CASHFLOW' AS table_name,
    COUNT(*) AS total_rows,
    CAST(NULL AS NUMBER) AS assets_n,
    CAST(NULL AS NUMBER) AS liabilities_n,
    CAST(NULL AS NUMBER) AS equity_incl_mi_n,
    CAST(NULL AS NUMBER) AS equity_excl_mi_n,
    CAST(NULL AS NUMBER) AS liabilities_equity_n,
    COUNT(NET_PROFIT) AS net_profit_incl_n,
    CAST(NULL AS NUMBER) AS net_profit_excl_n,
    CAST(NULL AS NUMBER) AS total_balance_profit_n,
    CAST(NULL AS NUMBER) AS ebit_n,
    CAST(NULL AS NUMBER) AS ebitda_n,
    COUNT(NET_CF_OP) AS operating_cashflow_n,
    COUNT(FREE_CASH_FLOW) AS free_cashflow_n
FROM MDB.IP_INST_FINC_CASHFLOW
WHERE DS_SRC = 'JY'
  AND RPT_TYPE = '11'
  AND ACTL_ANNC_DT <= 20250930;
