-- Issuer-rating time range
SELECT
    MIN(RT_DT) AS min_rt_dt,
    MAX(RT_DT) AS max_rt_dt,
    MIN(ANNC_DT) AS min_annc_dt,
    MAX(ANNC_DT) AS max_annc_dt,
    COUNT(*) AS rating_records,
    COUNT(DISTINCT INST_CODE) AS rated_institutions
FROM MDB.IV_BOND_INST_RATING;

-- Rating-change codes
SELECT
    RT_CHG,
    COUNT(*) AS event_count,
    COUNT(DISTINCT INST_CODE) AS institution_count
FROM MDB.IV_BOND_INST_RATING
GROUP BY RT_CHG
ORDER BY event_count DESC;

-- Rating types
SELECT
    RT_TYPE,
    COUNT(*) AS event_count,
    COUNT(DISTINCT INST_CODE) AS institution_count
FROM MDB.IV_BOND_INST_RATING
GROUP BY RT_TYPE
ORDER BY event_count DESC;

-- Default-data coverage and event counts
SELECT
    MIN(DEF_PUB_DT) AS min_default_pub_dt,
    MAX(DEF_PUB_DT) AS max_default_pub_dt,
    MIN(PAY_DT) AS min_pay_dt,
    MAX(PAY_DT) AS max_pay_dt,
    COUNT(*) AS raw_records,
    COUNT(DISTINCT DEF_ID) AS default_events,
    COUNT(DISTINCT SCR_ID) AS defaulted_bonds,
    COUNT(DISTINCT INST_ID) AS affected_institutions
FROM MDB.IV_BOND_CR_DEFAULT;

-- Decode RT_CHG
SELECT
    RT_CHG,
    PREV_RANK,
    RT_RANK,
    transition_count
FROM (
    SELECT
        x.*,
        ROW_NUMBER() OVER (
            PARTITION BY RT_CHG
            ORDER BY transition_count DESC
        ) AS rn
    FROM (
        SELECT
            RT_CHG,
            PREV_RANK,
            RT_RANK,
            COUNT(*) AS transition_count
        FROM MDB.IV_BOND_INST_RATING
        WHERE PREV_RANK IS NOT NULL
          AND RT_RANK IS NOT NULL
        GROUP BY
            RT_CHG,
            PREV_RANK,
            RT_RANK
    ) x
)
WHERE rn <= 15
ORDER BY RT_CHG, transition_count DESC;

-- Confirm rating-type meanings
-- Inspect database field comments
SELECT
    COLUMN_NAME,
    COMMENTS
FROM ALL_COL_COMMENTS
WHERE OWNER = 'MDB'
  AND TABLE_NAME = 'IV_BOND_INST_RATING'
  AND COLUMN_NAME IN (
      'RT_TYPE',
      'RT_CHG',
      'RT_RANK',
      'PREV_RANK',
      'RT_DT',
      'ANNC_DT'
  )
ORDER BY COLUMN_NAME;
-- Inspect representative records for common rating types
SELECT
    RT_TYPE,
    RT_CHG,
    RT_RANK,
    PREV_RANK,
    RT_INST_NAME,
    RT_MEMO
FROM (
    SELECT
        r.*,
        ROW_NUMBER() OVER (
            PARTITION BY RT_TYPE
            ORDER BY NVL(ANNC_DT, RT_DT) DESC
        ) AS rn
    FROM MDB.IV_BOND_INST_RATING r
    WHERE RT_TYPE IN ('3', '31', '34', '300', '36', '8')
)
WHERE rn <= 3
ORDER BY RT_TYPE;

-- Deduplicate by first default announcement
SELECT
    SUBSTR(TO_CHAR(first_default_dt), 1, 4) AS default_year,
    COUNT(*) AS first_default_institutions
FROM (
    SELECT
        INST_ID,
        MIN(DEF_PUB_DT) AS first_default_dt
    FROM MDB.IV_BOND_CR_DEFAULT
    WHERE INST_ID IS NOT NULL
      AND DEF_PUB_DT BETWEEN 19900101 AND 20260807
    GROUP BY INST_ID
)
GROUP BY SUBSTR(TO_CHAR(first_default_dt), 1, 4)
ORDER BY default_year;
-- Also summarize by bond
SELECT
    SUBSTR(TO_CHAR(first_default_dt), 1, 4) AS default_year,
    COUNT(*) AS first_default_bonds
FROM (
    SELECT
        SCR_ID,
        MIN(DEF_PUB_DT) AS first_default_dt
    FROM MDB.IV_BOND_CR_DEFAULT
    WHERE SCR_ID IS NOT NULL
      AND DEF_PUB_DT BETWEEN 19900101 AND 20260807
    GROUP BY SCR_ID
)
GROUP BY SUBSTR(TO_CHAR(first_default_dt), 1, 4)
ORDER BY default_year;

-- Check financial-data coverage
SELECT
    'BALANCE' AS source,
    MIN(ACTL_ANNC_DT) AS min_annc_dt,
    MAX(ACTL_ANNC_DT) AS max_annc_dt,
    COUNT(*) AS records,
    COUNT(DISTINCT INST_ID) AS institutions
FROM MDB.IP_INST_FINC_BALANCE
UNION ALL
SELECT
    'INCOME',
    MIN(ACTL_ANNC_DT),
    MAX(ACTL_ANNC_DT),
    COUNT(*),
    COUNT(DISTINCT INST_ID)
FROM MDB.IP_INST_FINC_INCOME
UNION ALL
SELECT
    'CASHFLOW',
    MIN(ACTL_ANNC_DT),
    MAX(ACTL_ANNC_DT),
    COUNT(*),
    COUNT(DISTINCT INST_ID)
FROM MDB.IP_INST_FINC_CASHFLOW;
