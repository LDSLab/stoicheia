use crate::catalog::{StorageConnection, StorageTransaction};
use crate::{Axis, BoundingBox, Fallible, Patch, PatchID, QuiltDetails, StoiError};
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

/// An implementation of tensor storage on SQLite
pub(crate) struct SQLiteConnection {
    conn: Mutex<rusqlite::Connection>,
}
impl SQLiteConnection {
    /// Create an in-memory SQLite database.
    ///
    /// Each connection creates a new database.
    pub fn connect_in_memory() -> Fallible<Arc<Self>> {
        Self::connect(":memory:".into())
    }

    /// Connect to an SQLite database
    ///
    /// SQLite treats the path ":memory:" as special and will only create an in-memory database
    /// in that case. See SQLite documentation for more details
    pub fn connect(base: PathBuf) -> Fallible<Arc<Self>> {
        let mut conn = rusqlite::Connection::open(base)?;
        {
            let txn = conn.transaction()?;
            txn.busy_timeout(std::time::Duration::from_secs(5))?;
            txn.execute_batch(include_str!("sqlite_catalog_schema.sql"))?;
            txn.commit()?;
        }
        Ok(Arc::new(Self {
            conn: Mutex::new(conn),
        }))
    }
}

impl<'t> StorageConnection for &'t SQLiteConnection {
    type Transaction = SQLiteTransaction<'t>;
    /// Create a new storage transaction on the database
    ///
    /// Most operations can only be done in a transaction, for correctness.
    fn txn(self) -> Fallible<SQLiteTransaction<'t>> {
        for i in 0..10 {
            if let Ok(txn) = self.conn.try_lock() {
                txn.execute_batch("BEGIN;")?;
                return Ok(SQLiteTransaction { txn });
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1 << i));
            }
        }
        Err(StoiError::RuntimeError(
            "sqlite mutex could not be acquired",
        ))
    }
}

pub(crate) struct SQLiteTransaction<'t> {
    txn: MutexGuard<'t, rusqlite::Connection>,
}
impl<'t> SQLiteTransaction<'t> {
    /// Put patch is only safe to do inside put_commit, so it's not part of Storage
    fn put_patch(&self, comm_id: i64, pat: &Patch) -> Fallible<PatchID> {
        let patch_id = PatchID(self.gen_id());
        self.txn.execute(
            "INSERT OR REPLACE INTO Patch(
                patch_id,
                comm_id,
                dim_0_min, dim_0_max,
                dim_1_min, dim_1_max,
                dim_2_min, dim_2_max,
                dim_3_min, dim_3_max
            ) VALUES (?,?,?,?,?,?,?,?,?,?);",
            &[
                &patch_id as &dyn ToSql,
                &comm_id,
                // TODO: HACK: Support real bounding boxes
                &0,
                &(1 << 30),
                &0,
                &(1 << 30),
                &0,
                &(1 << 30),
                &0,
                &(1 << 30),
            ],
        )?;
        // TODO: If this serialize fails it will deadlock the connection by not rolling back
        self.txn.execute(
            "INSERT OR REPLACE INTO PatchContent(patch_id, content) VALUES (?,?);",
            &[&patch_id as &dyn ToSql, &pat.compact().serialize(None)?],
        )?;
        Ok(patch_id)
    }

    /// Generate an id using the time plus a small salt
    fn gen_id(&self) -> i64 {
        chrono::Utc::now().timestamp_nanos() + rand::random::<i16>() as i64
    }
}

impl<'t> StorageTransaction for SQLiteTransaction<'t> {
    fn create_axis(&self, axis_name: &str, ignore_if_exists: bool) -> Fallible<()> {
        self.txn.execute(
            &format!(
                "INSERT {} INTO AxisContent(axis_name, content) VALUES (?,?);",
                if ignore_if_exists { "OR IGNORE" } else { "" }
            ),
            &[
                &axis_name as &dyn ToSql,
                &bincode::serialize(&Axis::empty(axis_name))?,
            ],
        )?;
        Ok(())
    }

    /// Replace a whole axis. Only do this through `Catalog.union_axis()`.
    fn put_axis(&self, axis: &Axis) -> Fallible<()> {
        self.txn.execute(
            "INSERT OR REPLACE INTO AxisContent(axis_name, content) VALUES (?,?);",
            &[&axis.name as &dyn ToSql, &bincode::serialize(&axis)?],
        )?;
        Ok(())
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, axis_name: &str) -> Fallible<Axis> {
        let res: Option<Vec<u8>> = self
            .txn
            .query_row(
                "SELECT content FROM AxisContent WHERE axis_name = ?",
                &[&axis_name],
                |r| r.get(0),
            )
            .optional()?;
        match res {
            None => Err(StoiError::NotFound("axis doesn't exist", axis_name.into())),
            Some(x) => Ok(bincode::deserialize(&x[..])?),
        }
    }

    /// Create a quilt
    fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()> {
        self.txn.execute(
            &format!(
                "INSERT {} INTO quilt(quilt_name, axes) VALUES (?, ?);",
                if ignore_if_exists { "OR IGNORE" } else { "" }
            ),
            &[&quilt_name, &serde_json::to_string(axes_names)?.as_ref()],
        )?;
        Ok(())
    }

    /// Get extended information about a quilt
    fn get_quilt_details(&self, quilt_name: &str) -> Fallible<QuiltDetails> {
        let deets = self
            .txn
            .query_row_and_then(
                "SELECT quilt_name, axes FROM quilt WHERE quilt_name = ?",
                &[&quilt_name],
                |r| QuiltDetails::try_from(r),
            )
            .optional()?;
        match deets {
            None => Err(StoiError::NotFound(
                "quilt doesn't exist",
                quilt_name.into(),
            )),
            Some(x) => Ok(x),
        }
    }

    /// List the currently available quilts
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>> {
        let mut map = HashMap::new();
        for row in self
            .txn
            .prepare("SELECT quilt_name, axes FROM quilt;")?
            .query_map(NO_PARAMS, |r| QuiltDetails::try_from(r))?
        {
            let row = row?;
            map.insert(row.name.clone(), row);
        }
        Ok(map)
    }

    /// Get the Patch IDs that would have to be applied to fill a fetch(), in the order they would
    /// need to be applied.
    ///
    /// You can specify many bounding boxes but the search degrades the more are provided. It's
    /// probably never a good idea to load more than a hundred.
    fn get_patches_by_bounding_boxes(
        &self,
        quilt_name: &str,
        tag: &str,
        bounding_boxes: &[BoundingBox],
    ) -> Fallible<Vec<PatchID>> {
        // This super long for with almost no body is admittedly rather strange here
        let mut patch_ids: Vec<PatchID> = vec![];
        for patch_id in self
            .txn
            .prepare(
                "
                WITH RECURSIVE CommitAncestry AS (
                    SELECT
                            comm_id parent_comm_id,
                            comm_id
                        FROM Tag
                        WHERE quilt_name = ?
                        AND tag_name = ?
                    UNION ALL
                    SELECT
                            Parent.parent_comm_id,
                            Parent.comm_id
                        FROM CommitAncestry Kid
                        INNER JOIN Comm Parent ON (Kid.parent_comm_id = Parent.comm_id)
                )
                SELECT
                    comm_id, patch_id
                    FROM CommitAncestry
                    INNER JOIN Patch USING (comm_id)
                    INNER JOIN json_each(?) BoundingBox ON (
                            dim_0_min <= json_extract(value, '$[0]')
                        AND dim_0_max >= json_extract(value, '$[1]')
                        AND dim_1_min <= json_extract(value, '$[2]')
                        AND dim_1_max >= json_extract(value, '$[3]')
                        AND dim_2_min <= json_extract(value, '$[4]')
                        AND dim_2_max >= json_extract(value, '$[5]')
                        AND dim_3_min <= json_extract(value, '$[6]')
                        AND dim_3_max >= json_extract(value, '$[7]')
                    )
                    GROUP BY comm_id, patch_id
                    ORDER BY comm_id ASC, patch_id ASC
            ",
            )?
            .query_map(
                &[
                    &quilt_name as &dyn ToSql,
                    &tag,
                    &serde_json::to_string(
                        &bounding_boxes
                            .iter()
                            .map(|bx| {
                                (0..4)
                                    .into_iter()
                                    .map(|ax_ix| bx.get(ax_ix).copied().unwrap_or((0, 1 << 30)))
                                    .collect_vec()
                            })
                            .map(|bx| {
                                [
                                    bx[0].0, bx[0].1, bx[1].0, bx[1].1, bx[2].0, bx[2].1, bx[3].0,
                                    bx[3].1,
                                ]
                            })
                            .collect_vec(),
                    )?,
                ],
                |r| r.get(1),
            )?
        {
            patch_ids.push(patch_id?);
        }
        Ok(patch_ids)
    }

    fn get_patch(&self, id: PatchID) -> Fallible<Patch> {
        let res: Vec<u8> = self.txn.query_row(
            "SELECT content FROM PatchContent WHERE patch_id = ?",
            &[&id],
            |r| r.get(0),
        )?;
        Ok(Patch::deserialize_from(&res[..])?)
    }

    // put_patch is part of Self, not Storage because you can only do it using put_commit()

    /// Push a commit to the database, balancing as necessary
    ///
    /// The heuristic used for balancing may change in the future, but this release works like so:
    ///
    ///     - Take patches from this commit that overlap this patch
    ///     - Fetch the area corresponding to the smallest one
    fn put_commit(
        &self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: Vec<&Patch>,
    ) -> Fallible<()> {
        let comm_id: i64 = self.gen_id();
        for pat in patches {
            self.put_patch(comm_id, pat)?;
        }
        self.txn.execute(
            // 1. Look for a tag given it's name and quilt.
            // 2. If there is one, get it's commit id and make that the parent commit ID to this one
            // 3. If there isn't one, then *still insert*, but with a null parent commit ID
            "INSERT INTO Comm(
                comm_id,
                parent_comm_id,
                message
            ) SELECT 
                ? comm_id,
                Parent.comm_id,
                ? message
            FROM (SELECT ? quilt_name, ? tag_name)
            LEFT JOIN Tag Parent USING (quilt_name, tag_name);",
            &[&comm_id as &dyn ToSql, &message, &quilt_name, &parent_tag],
        )?;
        self.txn.execute(
            "INSERT OR REPLACE INTO Tag(
                quilt_name,
                tag_name,
                comm_id
            ) VALUES (?, ?, ?)",
            &[&quilt_name as &dyn ToSql, &new_tag, &comm_id],
        )?;
        Ok(())
    }

    /// Commit the transaction within
    fn finish(self) -> Fallible<()> {
        Ok(self.txn.execute_batch("COMMIT;")?)
    }
}

/// Rollback the transaction by default
impl<'t> Drop for SQLiteTransaction<'t> {
    fn drop(&mut self) {
        self.txn.execute_batch("ROLLBACK;").unwrap_or(());
    }
}
