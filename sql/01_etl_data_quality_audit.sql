SELECT 'ETL_DQ_RULES' AS table_name, COUNT(*) AS row_count
FROM MDB.ETL_DQ_RULES
UNION ALL
SELECT 'ETL_DQ_RULES_RESULT', COUNT(*)
FROM MDB.ETL_DQ_RULES_RESULT
UNION ALL
SELECT 'ETL_DETAIL_LOG', COUNT(*)
FROM MDB.ETL_DETAIL_LOG
UNION ALL
SELECT 'ETL_TRANSFER_TIME_LOG', COUNT(*)
FROM MDB.ETL_TRANSFER_TIME_LOG
UNION ALL
SELECT 'ETL_TABLE_DS_CHECK', COUNT(*)
FROM MDB.ETL_TABLE_DS_CHECK;

-- Data-quality coverage
SELECT
    MIN(CHECK_DT) AS min_check_dt,
    MAX(CHECK_DT) AS max_check_dt,
    COUNT(DISTINCT CHECK_DT) AS check_days,
    COUNT(DISTINCT TABLE_NAME) AS checked_tables,
    COUNT(DISTINCT RULE_ID) AS executed_rules
FROM MDB.ETL_DQ_RULES_RESULT;

-- ETL log coverage
SELECT
    MIN(START_TIME) AS min_start_time,
    MAX(START_TIME) AS max_start_time,
    COUNT(DISTINCT ETL_NAME) AS etl_jobs,
    COUNT(*) AS executions
FROM MDB.ETL_DETAIL_LOG;

-- Status-code lookup
SELECT
    IS_SUCCESS,
    COUNT(*) AS result_count
FROM MDB.ETL_DQ_RULES_RESULT
GROUP BY IS_SUCCESS
ORDER BY result_count DESC;

-- Status-code lookup
SELECT
    STATUS,
    ERR_LEVEL,
    COUNT(*) AS execution_count
FROM MDB.ETL_DETAIL_LOG
GROUP BY STATUS, ERR_LEVEL
ORDER BY execution_count DESC;

-- Decode ETL status
SELECT
    STATUS,
    ERR_LEVEL,
    ETL_NAME,
    ETL_DESC,
    START_TIME,
    END_TIME,
    DATA_RECORDS,
    TIME_OUT,
    SUBSTR(ETL_MSG, 1, 300) AS etl_msg
FROM (
    SELECT
        l.*,
        ROW_NUMBER() OVER (
            PARTITION BY STATUS, ERR_LEVEL
            ORDER BY START_TIME DESC
        ) AS rn
    FROM MDB.ETL_DETAIL_LOG l
)
WHERE rn <= 3
ORDER BY STATUS, ERR_LEVEL, START_TIME DESC;

-- Resolve status codes precisely
-- Aggregated query
SELECT
    STATUS,
    ERR_LEVEL,
    COUNT(*) AS total_count,
    SUM(
        CASE
            WHEN UPPER(ETL_MSG) LIKE '%SUCCESS%'
            THEN 1 ELSE 0
        END
    ) AS success_message_count,
    SUM(
        CASE
            WHEN UPPER(ETL_MSG) LIKE '%ORA-%'
              OR UPPER(ETL_MSG) LIKE '%ETL.ERROR%'
              OR UPPER(ETL_MSG) LIKE '%ERROR%'
            THEN 1 ELSE 0
        END
    ) AS error_message_count,
    SUM(
        CASE
            WHEN ETL_MSG IS NULL
            THEN 1 ELSE 0
        END
    ) AS null_message_count
FROM MDB.ETL_DETAIL_LOG
GROUP BY STATUS, ERR_LEVEL
ORDER BY STATUS, ERR_LEVEL;

-- Check whether TIME_OUT records elapsed runtime
SELECT
    STATUS,
    ERR_LEVEL,
    COUNT(*) AS execution_count,
    ROUND(AVG(TIME_OUT), 2) AS avg_time_out,
    MIN(TIME_OUT) AS min_time_out,
    MAX(TIME_OUT) AS max_time_out,
    ROUND(
        AVG(
            CASE
                WHEN START_TIME IS NOT NULL
                 AND END_TIME IS NOT NULL
                THEN (END_TIME - START_TIME) * 24 * 60 * 60
            END
        ),
        2
    ) AS calculated_avg_seconds
FROM MDB.ETL_DETAIL_LOG
GROUP BY STATUS, ERR_LEVEL
ORDER BY STATUS, ERR_LEVEL;

-- Inspect the principal Oracle error types
SELECT
    CASE
        WHEN ETL_MSG LIKE '%ORA-%'
        THEN REGEXP_SUBSTR(ETL_MSG, 'ORA-[0-9]{5}')
        ELSE 'NO_ORA_CODE'
    END AS error_code,
    COUNT(*) AS occurrence_count,
    COUNT(DISTINCT ETL_NAME) AS affected_jobs
FROM MDB.ETL_DETAIL_LOG
WHERE UPPER(ETL_MSG) LIKE '%ERROR%'
   OR UPPER(ETL_MSG) LIKE '%ORA-%'
GROUP BY
    CASE
        WHEN ETL_MSG LIKE '%ORA-%'
        THEN REGEXP_SUBSTR(ETL_MSG, 'ORA-[0-9]{5}')
        ELSE 'NO_ORA_CODE'
    END
ORDER BY occurrence_count DESC;

-- Run the two previously unreported DQ queries to complete the DQ-failure structure analysis
-- Tables with the most failures
SELECT
    TABLE_NAME,
    COUNT(*) AS total_checks,
    SUM(CASE WHEN IS_SUCCESS = '0' THEN 1 ELSE 0 END)
        AS failed_checks,
    ROUND(
        100 * SUM(CASE WHEN IS_SUCCESS = '0' THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS failure_pct
FROM MDB.ETL_DQ_RULES_RESULT
GROUP BY TABLE_NAME
HAVING COUNT(*) >= 20
ORDER BY failed_checks DESC, failure_pct DESC;
-- Rules with the most failures
SELECT
    r.RULE_ID,
    q.NAME AS rule_name,
    r.TABLE_NAME,
    q.RULE_TYPE,
    COUNT(*) AS total_checks,
    SUM(CASE WHEN r.IS_SUCCESS = '0' THEN 1 ELSE 0 END)
        AS failed_checks,
    ROUND(
        100 * SUM(CASE WHEN r.IS_SUCCESS = '0' THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS failure_pct
FROM MDB.ETL_DQ_RULES_RESULT r
LEFT JOIN MDB.ETL_DQ_RULES q
    ON r.RULE_ID = q.ID
GROUP BY
    r.RULE_ID,
    q.NAME,
    r.TABLE_NAME,
    q.RULE_TYPE
HAVING COUNT(*) >= 10
ORDER BY failed_checks DESC, failure_pct DESC;
-- ETL tasks most prone to failure
SELECT
    ETL_NAME,
    MAX(ETL_DESC) AS etl_desc,
    COUNT(*) AS total_executions,
    SUM(
        CASE
            WHEN STATUS = '1'
              OR (STATUS = '-1' AND ERR_LEVEL = '2')
            THEN 1 ELSE 0
        END
    ) AS failed_executions,
    ROUND(
        100 * SUM(
            CASE
                WHEN STATUS = '1'
                  OR (STATUS = '-1' AND ERR_LEVEL = '2')
                THEN 1 ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS failure_pct,
    ROUND(AVG(TIME_OUT), 2) AS avg_runtime_seconds
FROM MDB.ETL_DETAIL_LOG
WHERE ERR_LEVEL = '2'
GROUP BY ETL_NAME
HAVING COUNT(*) >= 10
ORDER BY failed_executions DESC, failure_pct DESC;
