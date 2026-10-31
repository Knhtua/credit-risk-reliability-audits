-- Retrieve the official rating-type names
SELECT
    DICT_TYPE,
    CODE,
    NAME,
    MEMO,
    STATUS
FROM MDB.BIZ_DICTIONARY
WHERE UPPER(DICT_TYPE) LIKE '%RT_CHG%'
   OR UPPER(DICT_TYPE) LIKE '%RT_TYPE%'
   OR UPPER(MEMO) LIKE '%RT_CHG%'
   OR UPPER(MEMO) LIKE '%RT_TYPE%'
   OR NAME LIKE UNISTR('%\\8BC4\\7EA7%')
ORDER BY DICT_TYPE, ORDER_NO, CODE;

-- Count annual rating downgrades
SELECT
    SUBSTR(
        TO_CHAR(NVL(ANNC_DT, RT_DT)),
        1,
        4
    ) AS downgrade_year,
    COUNT(*) AS raw_downgrade_records,
    COUNT(DISTINCT INST_CODE) AS downgraded_institutions
FROM MDB.IV_BOND_INST_RATING
WHERE RT_CHG = '4'
  AND NVL(ANNC_DT, RT_DT) BETWEEN 19900101 AND 20260807
GROUP BY
    SUBSTR(
        TO_CHAR(NVL(ANNC_DT, RT_DT)),
        1,
        4
    )
ORDER BY downgrade_year;
-- Further split by rating type
SELECT
    RT_TYPE,
    COUNT(*) AS raw_downgrade_records,
    COUNT(DISTINCT INST_CODE) AS downgraded_institutions
FROM MDB.IV_BOND_INST_RATING
WHERE RT_CHG = '4'
GROUP BY RT_TYPE
ORDER BY raw_downgrade_records DESC;

-- Check whether rating and financial issuer IDs can be joined directly
WITH rated AS (
    SELECT DISTINCT INST_CODE
    FROM MDB.IV_BOND_INST_RATING
    WHERE INST_CODE IS NOT NULL
),
financial AS (
    SELECT DISTINCT INST_ID
    FROM MDB.IP_INST_FINC_BALANCE
    WHERE INST_ID IS NOT NULL
    UNION
    SELECT DISTINCT INST_ID
    FROM MDB.IP_INST_FINC_INCOME
    WHERE INST_ID IS NOT NULL
    UNION
    SELECT DISTINCT INST_ID
    FROM MDB.IP_INST_FINC_CASHFLOW
    WHERE INST_ID IS NOT NULL
)
SELECT
    COUNT(*) AS rated_institutions,
    SUM(
        CASE
            WHEN f.INST_ID IS NOT NULL
            THEN 1 ELSE 0
        END
    ) AS matched_institutions,
    ROUND(
        100 * SUM(
            CASE
                WHEN f.INST_ID IS NOT NULL
                THEN 1 ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS match_pct
FROM rated r
LEFT JOIN financial f
    ON r.INST_CODE = f.INST_ID;
    
-- Check announcement-date reliability
SELECT
    COUNT(*) AS total_n,
    SUM(CASE WHEN ANNC_DT IS NULL THEN 1 ELSE 0 END) AS miss_annc,
    SUM(CASE
        WHEN ANNC_DT IS NOT NULL
         AND RT_DT IS NOT NULL
         AND ANNC_DT < RT_DT
        THEN 1 ELSE 0
    END) AS annc_before_rt,
    SUM(CASE
        WHEN ANNC_DT IS NOT NULL
         AND RT_DT IS NOT NULL
         AND ANNC_DT >= RT_DT
        THEN 1 ELSE 0
    END) AS annc_ge_rt
FROM MDB.IV_BOND_INST_RATING;
-- Examine downgrade events specifically
SELECT
    COUNT(*) AS downgrade_records,
    SUM(CASE WHEN ANNC_DT IS NULL THEN 1 ELSE 0 END)
        AS missing_annc_dt,
    SUM(CASE WHEN PREV_RANK IS NULL OR TRIM(PREV_RANK) IS NULL
             THEN 1 ELSE 0 END)
        AS missing_previous_rank,
    COUNT(DISTINCT INST_CODE) AS institutions
FROM MDB.IV_BOND_INST_RATING
WHERE RT_CHG = '4';

-- Retrieve dictionary hierarchy fields
SELECT
    DS_ID,
    PRNT_ID,
    CODE,
    NAME,
    MEMO,
    DICT_TYPE,
    STATUS
FROM MDB.BIZ_DICTIONARY
WHERE MEMO IN (UNISTR('\\8BC4\\7EA7\\7C7B\\578B'), UNISTR('\\8BC4\\7EA7\\5BF9\\8C61\\7C7B\\578B'))
ORDER BY MEMO, PRNT_ID, CODE;
-- Inspect fields in the ETL dictionary-mapping table
SELECT
    COLUMN_ID,
    COLUMN_NAME,
    DATA_TYPE,
    DATA_LENGTH,
    COMMENTS
FROM (
    SELECT
        c.COLUMN_ID,
        c.COLUMN_NAME,
        c.DATA_TYPE,
        c.DATA_LENGTH,
        m.COMMENTS
    FROM ALL_TAB_COLUMNS c
    LEFT JOIN ALL_COL_COMMENTS m
      ON c.OWNER = m.OWNER
     AND c.TABLE_NAME = m.TABLE_NAME
     AND c.COLUMN_NAME = m.COLUMN_NAME
    WHERE c.OWNER = 'MDB'
      AND c.TABLE_NAME = 'ETL_DICT_MAPPING'
)
ORDER BY COLUMN_ID;

-- Query rating-type mappings
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
WHERE UPPER(DICT_CODE) = 'RT_TYPE'
   OR DICT_NAME LIKE UNISTR('%\\8BC4\\7EA7\\7C7B\\578B%')
ORDER BY
    DICT_ITEM_CODE,
    SRC_SYS,
    SRC_ITEM_CODE;

-- Determine the meanings of RT_TYPE=3, 31, 34, 300, and related codes in the final database table
SELECT
    DS_ID,
    PRNT_ID,
    CODE,
    NAME,
    MEMO,
    DICT_TYPE,
    STATUS
FROM MDB.BIZ_DICTIONARY
WHERE PRNT_ID = 'RT_TYPE'
ORDER BY
    CASE
        WHEN REGEXP_LIKE(CODE, '^[0-9]+$')
        THEN TO_NUMBER(CODE)
    END,
    CODE;

-- Data source + RT_TYPE + rating-type name + downgrade-event count
SELECT
    r.DS_SRC,
    r.RT_TYPE,
    NVL(m.DICT_ITEM_CODE, 'UNMAPPED') AS mapped_code,
    NVL(m.SRC_ITEM_NAME, UNISTR('\\672A\\627E\\5230\\6620\\5C04')) AS rating_type_name,
    COUNT(*) AS downgrade_records,
    COUNT(DISTINCT r.INST_CODE) AS downgraded_inst
FROM MDB.IV_BOND_INST_RATING r
LEFT JOIN MDB.ETL_DICT_MAPPING m
    ON m.DICT_CODE = 'RT_TYPE'
   AND m.SRC_SYS = r.DS_SRC
   AND m.SRC_ITEM_CODE = r.RT_TYPE
   AND m.STATUS = '1'
WHERE r.RT_CHG = '4'
GROUP BY
    r.DS_SRC,
    r.RT_TYPE,
    NVL(m.DICT_ITEM_CODE, 'UNMAPPED'),
    NVL(m.SRC_ITEM_NAME, UNISTR('\\672A\\627E\\5230\\6620\\5C04'))
ORDER BY
    downgrade_records DESC;
    
-- Deduplicate records with the same issuer and announcement date
SELECT
    SUBSTR(TO_CHAR(ANNC_DT), 1, 4) AS event_year,
    COUNT(*) AS raw_records,
    COUNT(
        DISTINCT INST_CODE || '|' || TO_CHAR(ANNC_DT)
    ) AS unique_events,
    COUNT(DISTINCT INST_CODE) AS event_inst
FROM MDB.IV_BOND_INST_RATING
WHERE DS_SRC = 'JY'
  AND RT_TYPE = '3'
  AND RT_CHG = '4'
  AND ANNC_DT IS NOT NULL
GROUP BY SUBSTR(TO_CHAR(ANNC_DT), 1, 4)
ORDER BY event_year;
