use crate::catalog::{StorageConnection, StorageTransaction};
use crate::{Axis, AxisSegment, BoundingBox, Fallible, Label, Patch, PatchID, PatchRef, QuiltDetails, StoiError};
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::{HashMap,HashSet};
use std::convert::{TryFrom, TryInto};
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
                return Ok(SQLiteTransaction { txn, axis_cache: HashMap::new() });
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
    axis_cache: HashMap<String, Axis>
}
impl<'t> SQLiteTransaction<'t> {
    /// Put patch is only safe to do inside put_commit, so it's not part of Storage
    fn put_patch(&self, comm_id: i64, pat: &Patch, bounding_box: BoundingBox) -> Fallible<PatchID> {
        let patch_id = PatchID(self.gen_id());
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
                &(pat.len() as i64),
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
            &[&patch_id as &dyn ToSql, &pat.serialize(None)?],
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
    fn put_axis(&mut self, axis: &Axis) -> Fallible<()> {
        self.txn.execute(
            "INSERT OR REPLACE INTO AxisContent(axis_name, content) VALUES (?,?);",
            &[&axis.name as &dyn ToSql, &bincode::serialize(&axis)?],
        )?;
        self.axis_cache.insert(axis.name.clone(), axis.clone());
        Ok(())
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&mut self, axis_name: &str) -> Fallible<&Axis> {
        if !self.axis_cache.contains_key(axis_name) {
            let res: Option<Vec<u8>> = self
            .txn
            .query_row(
                "SELECT content FROM AxisContent WHERE axis_name = ?",
                &[&axis_name],
                |r| r.get(0),
            )
            .optional()?;
            match res {
                None => return Err(StoiError::NotFound("axis doesn't exist", axis_name.into())),
                Some(x) => {
                    let axis : Axis = bincode::deserialize(&x[..])?;
                    self.axis_cache.insert(axis_name.to_string(), axis);
                },
            }
        }
        Ok(self.axis_cache.get(axis_name).unwrap())
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
    fn get_patches_by_bounding_boxes(
        &self,
        quilt_name: &str,
        tag: &str,
        deep: bool,
        bounding_boxes: &[BoundingBox],
    ) -> Fallible<Vec<PatchRef>> {
        // This is a fairly complex query we need to run so it deserved long-hand
        let mut stmt = self
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
            ")?;
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
                            bx[0].0, bx[0].1, bx[1].0, bx[1].1, bx[2].0, bx[2].1, bx[3].0,
                            bx[3].1,
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
                    (row.get::<usize, i64>(2)? as usize, row.get::<usize, i64>(3)? as usize),
                    (row.get::<usize, i64>(4)? as usize, row.get::<usize, i64>(5)? as usize),
                    (row.get::<usize, i64>(6)? as usize, row.get::<usize, i64>(7)? as usize),
                    (row.get::<usize, i64>(8)? as usize, row.get::<usize, i64>(9)? as usize),
                ],
            });
        }
        Ok(patch_refs)
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

    /// Get the bounding box of a patch
    ///
    /// These bounding boxes depend on the storage order of the catalog, so they aren't something
    /// the Patch could know on its own, instead you find this through the catalog
    fn get_bounding_box(&mut self, patch: &Patch) -> Fallible<BoundingBox> {
        let bbvec = (0..4)
            .map(|ax_ix| {
                match patch.axes().get(ax_ix) {
                    Some(patch_axis) => {
                        let patch_axis_labelset: HashSet<Label> =
                        patch_axis.labels().iter().copied().collect();
                        let global_axis = self.get_axis(&patch_axis.name)?;
                        let first = global_axis
                            .labels()
                            .iter()
                            .position(|x| patch_axis_labelset.contains(x));
                        let last = global_axis
                            .labels()
                            .iter()
                            .rposition(|x| patch_axis_labelset.contains(x));
    
                        Ok((first.unwrap_or(0), last.unwrap_or(1 << 60)))
                    }
                    None => Ok((0, 1<<60))
                }
                
            })
            .collect::<Fallible<Vec<AxisSegment>>>()?;
        Ok(bbvec[..].try_into()?)
    }

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
        patches: Vec<&Patch>,
    ) -> Fallible<()> {
        // The heuristic used for balancing may change in the future, but this is my suggestion:
        //
        //     - Take patches from this commit that overlap this patch
        //          - con: It leaves alone anything that doesn't overlap
        //     - Merge it with the smallest one
        //     - If it gets too large, split it by the longest dimension
        //
        let comm_id: i64 = self.gen_id();
        for pat in patches {
            let new_bounding_box = self.get_bounding_box(pat)?;
            
            // Find a friend to merge with: choosing the smallest will bring up the tiny patchlets
            let friend_patch_ref = self
                .get_patches_by_bounding_boxes(quilt_name, new_tag, false, &[new_bounding_box])?
                .into_iter()
                .min_by_key(|patch_ref| patch_ref.decompressed_size);
            
            // Find the visible area, not the original. If it was occluded by another (larger?) patch
            // in between, we need to include that occlusion in the new patch because it's what you
            // would have seen if you had fetch()ed
            //let friend_visible_area = self.fetch(quilt_name, new_tag)
            self.put_patch(comm_id, pat, new_bounding_box)?;
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
