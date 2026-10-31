SELECT
    USER AS current_user,
    SYSDATE AS database_time
FROM dual;

SELECT banner
FROM v$version;

SELECT COUNT(*) AS table_count
FROM user_tables;

SELECT
    table_name,
    num_rows,
    last_analyzed
FROM user_tables
ORDER BY table_name;

SELECT
    owner,
    COUNT(*) AS table_count
FROM all_tables
WHERE owner NOT IN (
    'SYS',
    'SYSTEM',
    'XDB',
    'MDSYS',
    'ORDSYS',
    'OUTLN',
    'DBSNMP'
)
GROUP BY owner
ORDER BY table_count DESC;

SELECT
    owner,
    table_name,
    num_rows,
    last_analyzed
FROM all_tables
WHERE owner NOT IN (
    'SYS',
    'SYSTEM',
    'XDB',
    'MDSYS',
    'ORDSYS',
    'OUTLN',
    'DBSNMP'
)
ORDER BY owner, table_name;

SELECT
    owner,
    table_name,
    column_id,
    column_name,
    data_type,
    data_length,
    data_precision,
    data_scale,
    nullable
FROM all_tab_columns
WHERE owner NOT IN (
    'SYS',
    'SYSTEM',
    'XDB',
    'MDSYS',
    'ORDSYS',
    'OUTLN',
    'DBSNMP'
)
ORDER BY owner, table_name, column_id;
