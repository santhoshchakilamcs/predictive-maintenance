CREATE TABLE IF NOT EXISTS sensor_readings (
    unit INTEGER NOT NULL,
    split TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    op_setting_1 REAL, op_setting_2 REAL, op_setting_3 REAL,
    sensor_1 REAL, sensor_2 REAL, sensor_3 REAL, sensor_4 REAL, sensor_5 REAL,
    sensor_6 REAL, sensor_7 REAL, sensor_8 REAL, sensor_9 REAL, sensor_10 REAL,
    sensor_11 REAL, sensor_12 REAL, sensor_13 REAL, sensor_14 REAL, sensor_15 REAL,
    sensor_16 REAL, sensor_17 REAL, sensor_18 REAL, sensor_19 REAL, sensor_20 REAL,
    sensor_21 REAL,
    PRIMARY KEY (split, unit, cycle)
);

CREATE TABLE IF NOT EXISTS rul_labels (
    split TEXT NOT NULL,
    unit INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    rul REAL NOT NULL,
    PRIMARY KEY (split, unit, cycle)
);

CREATE TABLE IF NOT EXISTS test_true_rul (
    unit INTEGER PRIMARY KEY,
    true_rul REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    unit INTEGER PRIMARY KEY,
    last_cycle INTEGER NOT NULL,
    true_rul REAL NOT NULL,
    pred_rul REAL NOT NULL,
    abs_error REAL NOT NULL,
    health_status TEXT NOT NULL,
    model_version TEXT NOT NULL
);
