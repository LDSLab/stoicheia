use failure::Fallible;
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thread_local::CachedThreadLocal;

use crate::{Axis, Label, Patch, PatchID, Quilt, QuiltMeta};

pub trait Catalog: Send + Sync {
    /// Get a quilt by name
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt>;

    /// Get only the metadata associated with a quilt by name
    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta>;

    /// Create a new quilt
    fn put_quilt(&self, meta: QuiltMeta) -> Fallible<()>;

    /// List all the quilts in the catalog
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>>;

    /// List all the patches that intersect a bounding box
    ///
    /// There may be false positives; some patches may not actually overlap
    /// There are no false negatives; all patches that overlap will be returned
    ///
    /// This method exists in case the database supports efficient multidimensional range queries
    /// such as SQLite or Postgres/PostGIS
    fn get_patches_by_bounding_box(
        &self,
        quilt_name: &str,
        bound: &[std::ops::Range<Label>],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>>;

    /// Get a single patch by ID
    fn get_patch(&self, id: PatchID) -> Fallible<Option<Patch<f32>>>;

    /// Save a single patch by ID
    fn put_patch(&self, id: PatchID, pat: Patch<f32>) -> Fallible<()>;

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis>;

    /// Replace the labels of an axis, in the order you would expect them to be stored
    fn put_axis(&self, name: &str, axis: Axis) -> Fallible<()>;
}

/// An in-memory catalog, meant for testing and dummy databases
pub struct MemoryCatalog {
    quilts: Mutex<HashMap<String, QuiltMeta>>,
    patches: Mutex<HashMap<PatchID, Patch<f32>>>,
    axes: Mutex<HashMap<String, Axis>>,
}
impl MemoryCatalog {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            quilts: Mutex::from(HashMap::new()),
            patches: Mutex::from(HashMap::new()),
            axes: Mutex::from(HashMap::new()),
        })
    }
}
impl Catalog for MemoryCatalog {
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt> {
        // Make sure it exists
        self.get_quilt_meta(quilt_name)?;
        Ok(Quilt::new(quilt_name.into(), self))
    }

    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta> {
        self.quilts
            .lock()
            .expect("Memory catalog is corrupted.")
            .get(quilt_name)
            .ok_or(format_err!("No such quilt {}", quilt_name))
            .cloned()
    }
    fn put_quilt(&self, meta: QuiltMeta) -> Fallible<()> {
        self.quilts
            .lock()
            .expect("Memory catalog is corrupted.")
            .insert(meta.name.clone(), meta);
        Ok(())
    }
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>> {
        Ok(self
            .quilts
            .lock()
            .expect("Memory catalog is corrupted")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }

    fn get_patches_by_bounding_box(
        &self,
        _quilt_name: &str,
        _bound: &[std::ops::Range<Label>],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>> {
        // This stub is hilariously memory inefficient but it's intended mostly for testing
        Ok(Box::new(
            self.patches
                .lock()
                .expect("Memory catalog is corrupted")
                .keys()
                .copied()
                // Copy, collect, and reiterate so that we can avoid keeping the mutex locked
                .collect_vec()
                .into_iter(),
        ))
    }

    fn get_patch(&self, id: PatchID) -> Fallible<Option<Patch<f32>>> {
        Ok(self
            .patches
            .lock()
            .expect("Memory catalog is corrupted")
            .get(&id)
            .cloned())
    }
    fn put_patch(&self, id: PatchID, pat: Patch<f32>) -> Fallible<()> {
        self.patches
            .lock()
            .expect("Memory catalog is corrupted")
            .insert(id, pat);
        Ok(())
    }

    fn get_axis(&self, axis_name: &str) -> Fallible<Axis> {
        self.axes
            .lock()
            .expect("Memory catalog is corrupted.")
            .get(axis_name)
            .ok_or(format_err!("No such quilt {}", axis_name))
            .cloned()
    }
    fn put_axis(&self, name: &str, axis: Axis) -> Fallible<()> {
        self.axes
            .lock()
            .expect("Memory catalog is corrupted.")
            .insert(name.into(), axis);
        Ok(())
    }
}

/// List of available tensors
pub struct SQLiteCatalog {
    base: PathBuf,
    conn: CachedThreadLocal<rusqlite::Connection>,
}
impl SQLiteCatalog {
    /// Create a shared in-memory SQLite database
    pub fn connect_in_memory() -> Fallible<Arc<Self>> {
        Ok(Arc::new(Self {
            base: "file::memory:?cache=shared".into(),
            conn: CachedThreadLocal::new(),
        }))
    }

    /// Connect to the underlying SQLite database
    pub fn connect(base: PathBuf) -> Fallible<Arc<Self>> {
        Ok(Arc::new(Self {
            base,
            conn: CachedThreadLocal::new(),
        }))
    }

    /// Gets the thread-local SQLite connection
    ///
    /// There is a separate connection for each thread because it saves time connecting,
    /// but it can't be global because they can't be used from different threads simultaneously
    fn get_conn(&self) -> Fallible<&rusqlite::Connection> {
        self.conn.get_or_try(|| {
            let conn = rusqlite::Connection::open(&self.base)?;
            conn.execute(
                "
                CREATE TABLE IF NOT EXISTS quilt(
                    quilt_name TEXT PRIMARY KEY COLLATE NOCASE,
                    axes TEXT NOT NULL CHECK (json_valid(axes))
                ) WITHOUT ROWID;
            ",
                NO_PARAMS,
            )?;

            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS patch_rtree(
                    patch_id INTEGER PRIMARY KEY,
                    dim_1_min, dim_1_max,
                    dim_2_min, dim_2_max,
                    dim_3_min, dim_3_max,
                    +quilt_name TEXT NOT NULL COLLATE NOCASE
                );",
                NO_PARAMS,
            )?;

            conn.execute(
                "
                CREATE TABLE IF NOT EXISTS patch_content(
                    patch_id INTEGER PRIMARY KEY,
                    content BLOB
                );
            ",
                NO_PARAMS,
            )?;

            conn.execute(
                "
                CREATE TABLE IF NOT EXISTS axis_content(
                    axis_name TEXT PRIMARY KEY,
                    content BLOB
                );
            ",
                NO_PARAMS,
            )?;
            Ok(conn)
        })
    }
}

impl Catalog for SQLiteCatalog {
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt> {
        // Make sure it exists
        self.get_quilt_meta(quilt_name)?;
        Ok(Quilt::new(quilt_name.into(), self))
    }
    /// Get extended information about a quilt
    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta> {
        Ok(self.get_conn()?.query_row_and_then(
            "SELECT quilt_name, axes FROM quilt WHERE quilt_name = ?",
            &[&quilt_name],
            |r| QuiltMeta::try_from(r),
        )?)
    }

    /// List the currently available quilts
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>> {
        let mut map = HashMap::new();
        for row in self
            .get_conn()?
            .prepare("SELECT quilt_name, axes FROM quilt;")?
            .query_map(NO_PARAMS, |r| QuiltMeta::try_from(r))?
        {
            let row = row?;
            map.insert(row.name.clone(), row);
        }
        Ok(map)
    }

    fn put_quilt(&self, meta: QuiltMeta) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT INTO quilt(quilt_name, axes) VALUES (?, ?);",
            &[&meta.name, &serde_json::to_string(&meta.axes)?],
        )?;
        Ok(())
    }

    fn get_patches_by_bounding_box(
        &self,
        quilt_name: &str,
        bound: &[std::ops::Range<Label>],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>> {
        // TODO: Verify that the dimensions match what we see in the quilt
        // The SQL formatting happens here because patch_tree
        let bound = (0..3)
            .map(|i| bound.get(i).cloned().unwrap_or(-1000000000..1000000000))
            .collect_vec();

        // Fetch patch ID's first, and then get them one by one. This is so we don't concurrently have multiple connections open.
        let mut patch_ids: Vec<i64> = vec![];
        for patch_id in self
            .get_conn()?
            .prepare(
                "SELECT patch_id
            FROM patch_rtree
            WHERE
                dim_0_min <= ?
                AND dim_0_max >= ?
                AND dim_1_min <= ?
                AND dim_1_max >= ?
                AND dim_2_min <= ?
                AND dim_2_max >= ?
                AND quilt_name = ?
            ",
            )?
            .query_map(
                &[
                    &bound[0].end as &dyn ToSql,
                    &bound[0].start,
                    &bound[1].end,
                    &bound[1].start,
                    &bound[2].end,
                    &bound[2].start,
                    &quilt_name,
                ],
                |r| r.get(0),
            )?
        {
            patch_ids.push(patch_id?);
        }

        Ok(Box::new(patch_ids.into_iter()))
    }

    fn get_patch(&self, id: PatchID) -> Fallible<Option<Patch<f32>>> {
        let res: Option<Vec<u8>> = self
            .get_conn()?
            .query_row(
                "SELECT content FROM patch WHERE patch_id = ?",
                &[&id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match res {
            None => None,
            Some(x) => bincode::deserialize(&x[..])?,
        })
    }

    fn put_patch(&self, id: PatchID, pat: Patch<f32>) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT OR REPLACE INTO patch(id, content) VALUES (?,?);",
            &[&id as &dyn ToSql, &bincode::serialize(&pat)?],
        )?;
        Ok(())
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis> {
        let res: Vec<u8> = self.get_conn()?.query_row(
            "SELECT content FROM axis_content WHERE axis_name = ?",
            &[&name],
            |r| r.get(0),
        )?;
        Ok(bincode::deserialize(&res[..])?)
    }

    /// Replace the labels of an axis, in the order you would expect them to be stored
    fn put_axis(&self, name: &str, axis: Axis) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT OR REPLACE INTO axis_content(axis_name, content) VALUES (?,?);",
            &[&name as &dyn ToSql, &bincode::serialize(&axis)?],
        )?;
        Ok(())
    }
}
