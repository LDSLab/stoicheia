use crate::catalog::{StorageConnection, StorageTransaction};
use crate::patch::PatchCompressionType;
use crate::{
    Axis, AxisSelection, BoundingBox, Counter, Fallible, Patch, PatchID, PatchRef, QuiltDetails,
    StoiError,
};
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use enum_map::EnumMap;

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
        let conn = rusqlite::Connection::open(base)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(include_str!("sqlite_catalog_schema.sql"))?;
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
                return Ok(SQLiteTransaction {
                    txn,
                    axis_cache: HashMap::new(),
                    trace: EnumMap::new(),
                });
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1 << i));
            }
        }
        Err(StoiError::RuntimeError(
            "sqlite mutex could not be acquired",
        ))
    }
}

/// A single database transaction.
///
/// Transactions are somewhat expensive, as they do incur disk activity,
/// and they also can act as critial sections, since in most cases only one write transaction
/// can be open at a time.
/// To maintain good performance, try to:
///  - Minimize transactions
///  - Begin() late and commit() promptly
///  - Delay the first write, (e.g. R-R-W-W, rather than R-W-R-W)
///  - Keep mutations per transaction between 16KB and 100MB
///
/// Keep in mind that even if your code is correct, transactions that mutate the database
/// can fail and may need to be retried. This happens if two transactions enter a race
/// condition.
#[derive(Debug)]
pub struct SQLiteTransaction<'t> {
    txn: MutexGuard<'t, rusqlite::Connection>,
    axis_cache: HashMap<String, Axis>,
    trace: EnumMap<Counter, usize>,
}
impl<'t> SQLiteTransaction<'t> {
    /// Put patch is only safe to do inside put_commit, so it's not part of Storage
    fn put_patch(
        &mut self,
        comm_id: i64,
        pat: &Patch,
        bounding_box: BoundingBox,
    ) -> Fallible<PatchID> {
        self.trace(Counter::WritePatch, 1);
        let patch_id = PatchID(self.gen_id());
        // Note - you need to compact here, quite late, because it needs to be after the Axes are updated.
        // That's because
        let pat = pat.compact();
        self.txn.execute(
            "INSERT OR REPLACE INTO Patch(
                patch_id,
                comm_id,
                decompressed_size,
                dim_0_min, dim_0_max,
                dim_1_min, dim_1_max,
                dim_2_min, dim_2_max,
                dim_3_min, dim_3_max
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?);",
            &[
                &patch_id as &dyn ToSql,
                &comm_id,
                &(4 * pat.len() as i64),
                &(bounding_box[0].0 as i64),
                &(bounding_box[0].1 as i64),
                &(bounding_box[1].0 as i64),
                &(bounding_box[1].1 as i64),
                &(bounding_box[2].0 as i64),
                &(bounding_box[2].1 as i64),
                &(bounding_box[3].0 as i64),
                &(bounding_box[3].1 as i64),
            ],
        )?;
        // TODO: If this serialize fails it will deadlock the connection by not rolling back
        self.txn.execute(
            "INSERT OR REPLACE INTO PatchContent(patch_id, content) VALUES (?,?);",
            &[
                &patch_id as &dyn ToSql,
                &pat.serialize(Some(PatchCompressionType::LZ4 { quality: 0 }))?,
            ],
        )?;
        Ok(patch_id)
    }

    /// Delete a patch
    ///
    /// This only makes sense untagg(), and for compaction as part of a commit(),
    /// and you can't do this from outside

    fn del_patch(&mut self, patch_id: PatchID) -> Fallible<()> {
        self.trace(Counter::DeletePatch, 1);
        self.txn
            .execute("DELETE FROM Patch WHERE patch_id = ?;", &[patch_id])?;
        self.txn
            .execute("DELETE FROM PatchContent WHERE patch_id = ?;", &[patch_id])?;
        Ok(())
    }

    /// Generate an id using the time plus a small salt
    fn gen_id(&self) -> i64 {
        chrono::Utc::now().timestamp_nanos() + rand::random::<i16>() as i64
    }
}

impl<'t> StorageTransaction for SQLiteTransaction<'t> {
    /// Increment an activity counter, used for performance and correctness checking
    fn trace(&mut self, ctr: Counter, increment: usize) {
        self.trace[ctr] += increment;
    }

    /// Retrieve performance counters, useful for debugging performance problems
    ///
    /// Returns: a Map containing the counters by name
    fn get_performance_counters(&self) -> EnumMap<Counter, usize> {
        self.trace.clone()
    }

    /// Append labels to an axis, in the order you would expect them to be stored.
    /// Any duplicate labels will not be appended.
    ///
    /// Returns true iff the axis was mutated in the process

    fn union_axis(&mut self, axis: &Axis) -> Fallible<bool> {
        let existing_labels = self.get_axis(&axis.name)?.labelset();

        let mut changes = 0;
        let mut trials = 0;
        changes += self.txn.execute(
            "INSERT OR IGNORE INTO Axis(axis_name) VALUES (?)",
            &[&axis.name],
        )?;
        let mut stmt = self
            .txn
            .prepare("INSERT OR IGNORE INTO AxisContent(axis_name, label) VALUES (?,?);")?;
        for label in axis.labels() {
            trials += 1;
            if !existing_labels.contains(label) {
                changes += stmt.execute(&[&axis.name as &dyn ToSql, &label])?;
            }
        }
        // Drop an immutable borrow so we can trace
        std::mem::drop(stmt);
        if changes > 0 {
            // Repair the cache
            self.axis_cache.get_mut(&axis.name).unwrap().union(&axis);
            self.trace(Counter::WriteAxisLabel, changes);
            self.trace(Counter::TrialAxisLabel, trials);
        }
        Ok(changes > 0)
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&mut self, axis_name: &str) -> Fallible<&Axis> {
        if !self.axis_cache.contains_key(axis_name) {
            self.trace(Counter::ReadAxis, 1);
            let mut stmt = self.txn.prepare(
                "SELECT label FROM AxisContent WHERE axis_name = ? ORDER BY global_storage_index",
            )?;
            let rows = stmt.query_map(&[&axis_name], |r| r.get::<_, i64>(0))?;
            let mut labels = vec![];
            for label in rows {
                labels.push(label?);
            }
            self.axis_cache
                .insert(axis_name.to_string(), Axis::new(axis_name, labels)?);
        }
        Ok(self.axis_cache.get(axis_name).unwrap())
    }

    /// List the currently available quilts
    fn list_quilts(&mut self) -> Fallible<HashMap<String, QuiltDetails>> {
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

    /// Create a quilt, and create axes as necessary to make it.
    fn create_quilt(&mut self, quilt_name: &str, axes_names: &[&str]) -> Fallible<bool> {
        let changes = self.txn.execute(
            "INSERT OR IGNORE INTO quilt(quilt_name, axes) VALUES (?, ?);",
            &[&quilt_name, &serde_json::to_string(axes_names)?.as_ref()],
        )?;
        Ok(changes > 0)
    }

    /// Get details about a quilt by name
    ///
    /// What details are available may depend on the quilt, and fields are likely to
    /// be added with time (so be careful with serializing QuiltDetails)
    fn get_quilt_details(&mut self, quilt_name: &str) -> Fallible<QuiltDetails> {
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

    /// Get the Patch IDs that would have to be applied to fill a fetch(), in the order they would
    /// need to be applied.
    ///
    /// You can specify many bounding boxes but the search degrades the more are provided. It's
    /// probably never a good idea to load more than a hundred.
    ///
    /// For fetching, you need deep=true, so you see the state of the tensor at that commit,
    /// but for compaction, deep=false can avoid filling in void areas.
    ///
    /// Accepts:
    ///     quilt_name: the quilt associated with the tag we're looking for
    ///     tag: the named commit
    ///     deep: whether to provide only the patches within this quilt, or all it's ancestors
    ///     bounding_boxes: returned patches should intersect at least one of these boxes
    ///
    /// Returns:
    ///     A vector of Patch ID's
    fn search(
        &mut self,
        quilt_name: &str,
        tag: &str,
        deep: bool,
        bounding_boxes: &[BoundingBox],
    ) -> Fallible<Vec<PatchRef>> {
        self.trace(Counter::SearchPatches, 1);
        // This is a fairly complex query we need to run so it deserved long-hand
        let mut stmt = self.txn.prepare(
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
                        INNER JOIN Comm Parent ON (? AND Kid.parent_comm_id = Parent.comm_id)
                )
                SELECT
                    patch_id, decompressed_size,
                    dim_0_min, dim_0_max,
                    dim_1_min, dim_1_max,
                    dim_2_min, dim_2_max,
                    dim_3_min, dim_3_max
                    FROM CommitAncestry
                    INNER JOIN Patch USING (comm_id)
                    INNER JOIN json_each(?) BoundingBox ON (
                            dim_0_max >= json_extract(value, '$[0]')
                        AND dim_0_min <= json_extract(value, '$[1]')
                        AND dim_1_max >= json_extract(value, '$[2]')
                        AND dim_1_min <= json_extract(value, '$[3]')
                        AND dim_2_max >= json_extract(value, '$[4]')
                        AND dim_2_min <= json_extract(value, '$[5]')
                        AND dim_3_max >= json_extract(value, '$[6]')
                        AND dim_3_min <= json_extract(value, '$[7]')
                    )
                    GROUP BY comm_id, patch_id
                    ORDER BY comm_id ASC, patch_id ASC
            ",
        )?;
        let mut rows = stmt.query(&[
            &quilt_name as &dyn ToSql,
            &tag,
            &deep, // This flag will enable/disable ancestor search
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
                            bx[0].0, bx[0].1, bx[1].0, bx[1].1, bx[2].0, bx[2].1, bx[3].0, bx[3].1,
                        ]
                    })
                    .collect_vec(),
            )?,
        ])?;

        let mut patch_refs: Vec<PatchRef> = vec![];
        while let Some(row) = rows.next()? {
            patch_refs.push(PatchRef {
                id: row.get(0)?,
                decompressed_size: row.get::<usize, i64>(1)? as u64,
                bounding_box: [
                    (
                        row.get::<usize, i64>(2)? as usize,
                        row.get::<usize, i64>(3)? as usize,
                    ),
                    (
                        row.get::<usize, i64>(4)? as usize,
                        row.get::<usize, i64>(5)? as usize,
                    ),
                    (
                        row.get::<usize, i64>(6)? as usize,
                        row.get::<usize, i64>(7)? as usize,
                    ),
                    (
                        row.get::<usize, i64>(8)? as usize,
                        row.get::<usize, i64>(9)? as usize,
                    ),
                ],
            });
        }
        Ok(patch_refs)
    }

    fn get_patch(&mut self, id: PatchID) -> Fallible<Patch> {
        self.trace(Counter::ReadPatch, 1);
        let res: Vec<u8> = self.txn.query_row(
            "SELECT content FROM PatchContent WHERE patch_id = ?",
            &[&id],
            |r| r.get(0),
        )?;
        self.trace(Counter::ReadBytes, res.len());
        let p = Patch::deserialize_from(&res[..])?;
        Ok(p)
    }

    // put_patch is part of Self, not Storage because you can only do it using put_commit()

    /// Make changes to a tensor via a commit
    ///
    /// This is only available together, so that the underlying storage media can do this
    /// atomically without a complicated API

    fn put_commit(
        &mut self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: &[&Patch],
    ) -> Fallible<()> {
        self.trace(Counter::PutCommit, 1);
        // The heuristic used for balancing may change in the future, but this is my suggestion:
        //
        //     - Take patches from this commit that overlap this patch
        //          - con: It leaves alone anything that doesn't overlap
        //     - Merge it with the smallest one
        //     - If it gets too large, split it by the longest dimension
        //
        let comm_id: i64 = self.gen_id();
        let mut pending_patches = vec![];
        for &pat in patches {
            let new_bounding_box = self.get_bounding_box(&pat)?;
            // Find a friend to merge with: choosing the smallest will bring up the tiny patchlets
            let maybe_friend_patch_ref = self
                .search(quilt_name, new_tag, false, &[new_bounding_box])?
                .into_iter()
                // TODO: Consider percent overlap
                .min_by_key(|patch_ref| patch_ref.decompressed_size);
            pending_patches.extend(match maybe_friend_patch_ref {
                Some(friend_patch_ref) => {
                    // Find the visible area, not just the original. If it was occluded by another (larger?) patch
                    // in between, we need to include that occlusion in the new patch because it's what you
                    // would have seen if you had fetch()ed
                    //
                    // We get the friend first because counter-intuitively, it's faster.
                    // In most cases the friend will not cover it's whole bounding box so it's
                    // much more efficient to create a selection from the friend instead.
                    self.trace(Counter::PutCommitGetPatch, 1);
                    let friend = self.get_patch(friend_patch_ref.id)?;
                    let patch_request = friend
                        .axes()
                        .iter()
                        .map(|ax| AxisSelection::Labels(ax.labels().to_vec()))
                        .collect_vec();
                    self.trace(Counter::PutCommitFetch, 1);
                    let friend_visible_area = self.fetch(quilt_name, new_tag, patch_request)?;
                    // Garbage collect the old patch because now it has been compacted into the new one
                    self.del_patch(friend_patch_ref.id)?;

                    // Merge the patch with it's friend
                    let new_large_patch = friend_visible_area.merge(&pat)?;
                    self.maybe_split(new_large_patch)
                }
                // TODO: Look at this clone
                None => Ok(vec![pat.to_owned()]),
            }?);
        }
        for new_patch in pending_patches {
            if new_patch.len() > 0 {
                // Add each new patch
                let bbox = self.get_bounding_box(&new_patch)?;
                self.put_patch(comm_id, &new_patch, bbox)?;
            }
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

    /// Commit the transaction
    fn finish(self) -> Fallible<()> {
        println!("Transaction completed with stats {:#?}", self.trace);
        Ok(self.txn.execute_batch("COMMIT;")?)
    }

    /// Rollback the transaction
    fn rollback(self) -> Fallible<()> {
        println!("Transaction rolled back with stats {:#?}", self.trace);
        Ok(self.txn.execute_batch("ROLLBACK;")?)
    }
}

/// Rollback the transaction by default
impl<'t> Drop for SQLiteTransaction<'t> {
    fn drop(&mut self) {
        self.txn.execute_batch("ROLLBACK;").unwrap_or(());
    }
}
