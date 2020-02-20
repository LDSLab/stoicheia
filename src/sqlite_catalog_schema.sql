-- This script gets run every time a new connection is made, so it should be idempotent.

CREATE TABLE IF NOT EXISTS Quilt(
    quilt_name TEXT COLLATE NOCASE PRIMARY KEY,
    axes       TEXT                NOT NULL CHECK (json_valid(axes))
) WITHOUT ROWID;

-- Later see if an r-tree actually changes performance
CREATE TABLE IF NOT EXISTS Patch (
    patch_id INTEGER PRIMARY KEY,
    comm_id  INTEGER NOT NULL REFERENCES Comm(comm_id),
    dim_0_min, dim_0_max,
    dim_1_min, dim_1_max,
    dim_2_min, dim_2_max,
    dim_3_min, dim_3_max
);

CREATE TABLE IF NOT EXISTS PatchContent(
    patch_id INTEGER PRIMARY KEY,
    content  BLOB
);

CREATE TABLE IF NOT EXISTS AxisContent(
    axis_name TEXT PRIMARY KEY,
    content   BLOB
);

CREATE TABLE IF NOT EXISTS Comm(
    comm_id        INTEGER PRIMARY KEY,
    parent_comm_id INTEGER                         REFERENCES Comm(comm_id),
    message TEXT
);

CREATE TABLE IF NOT EXISTS Tag(
    quilt_name TEXT COLLATE NOCASE REFERENCES Quilt(quilt_name),
    tag_name   TEXT COLLATE NOCASE,
    comm_id INTEGER NOT NULL REFERENCES Comm(comm_id),

    PRIMARY KEY (quilt_name, tag_name)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS Tag__comm_id ON Tag(comm_id);