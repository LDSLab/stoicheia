use crate::Fallible;
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;
use thread_local::CachedThreadLocal;

use crate::{Axis, BoundingBox, Patch, PatchID, Quilt, QuiltMeta, AxisSelection, AxisSegment, PatchRequest, StoiError};

pub trait Catalog: Send + Sync {
    /// Get a quilt by name
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt>;

    /// Get only the metadata associated with a quilt by name
    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta>;

    /// Create a new quilt
    fn create_quilt(&self, quilt_name: &str, axes_names: &[&str]) -> Fallible<()>;

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
        bound: &BoundingBox,
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>>;

    /// Get a single patch by ID
    fn get_patch(&self, id: PatchID) -> Fallible<Patch<f32>>;

    /// Save a single patch by ID
    fn put_patch(&self, id: PatchID, pat: Patch<f32>) -> Fallible<()>;

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis>;

    /// Replace the labels of an axis, in the order you would expect them to be stored
    fn union_axis(&self, axis: &Axis) -> Fallible<()>;

    ///
    /// Default implementations
    /// 

    /// Resolve the labels that a patch request selects
    fn get_axis_from_selection(&self, sel: AxisSelection) -> Fallible<(Axis, Vec<AxisSegment>)> {
        Ok(match sel {
            AxisSelection::All { name } => {
                let axis = self.get_axis(&name)?;
                let full_range = 0..=axis.len();
                (axis, vec![full_range])
            }
            AxisSelection::Labels { name, labels } => {
                // TODO: Profile this - it could be a performance issue
                let axis: Axis = self.get_axis(&name)?;
                let labelset = axis.labelset();
                let start_ix = axis
                    .labels()
                    .iter()
                    .position(|x| labelset.contains(&x))
                    .unwrap_or(axis.len());
                let end_ix = axis.labels()[start_ix..]
                    .iter()
                    .rposition(|x| labelset.contains(&x))
                    .unwrap_or(0);
                (Axis::new(name, labels)?, vec![start_ix..=end_ix])
            }
            AxisSelection::RangeInclusive { name, start, end } => {
                // Axis labels are not guaranteed to be sorted because it may be optimized for storage, not lookup
                let axis: Axis = self.get_axis(&name)?;
                let lab = axis.labels();
                let start_ix = lab
                    .iter()
                    .position(|&x| x == start)
                    // If we can't find that label we don't search anything
                    .unwrap_or(axis.len());
                let end_ix = start_ix
                    + lab[start_ix..]
                        .iter()
                        .position(|&x| x == end)
                        .unwrap_or(axis.len() - start_ix);
                (
                    Axis::new(&axis.name, Vec::from(&lab[start_ix..=end_ix]))?,
                    vec![start_ix..=end_ix],
                )
            }
        })
    }

    /// Assemble a patch from a quilt
    ///
    /// The tensor name is part of the patch so it doesn't need to be specified
    fn fetch(&self, quilt_name: &str, request: PatchRequest) -> Fallible<Patch<f32>> {
        if request.is_empty() {
            return Err(StoiError::InvalidValue(
                "Patches must have at least one axis",
            ));
        }

        //
        // Find all the labels of the axes they are planning to use
        //

        // Names and all labels of all of the axes involved
        let mut axes = vec![];
        // Segments of each axis, which will be the edges of bounding boxes
        let mut segments_by_axis = vec![];
        for sel in request {
            let (axis, segments) = self.get_axis_from_selection(sel)?;
            axes.push(axis);
            segments_by_axis.push(segments);
        }
        // At this point we know how big the output will be.
        // The error here is early to avoid the IO
        // and we don't construct the patch (which would have noticed and raised the same error)
        // in order to avoid holding memory longer
        if axes.iter().map(|a| a.len()).product::<usize>() > 256 << 20 {
            return Err(StoiError::TooLarge(
                "Patches must be 256 million elements or less (1GB of 32bit floats)",
            ));
        }

        //
        // Find all bounding boxes we need to get the cartesian product of all the axis segments
        //

        // If there are more than 1000 bounding boxes, collapse them, to protect the R*tree (or whatever index) from DOS
        let total_bounding_boxes: usize = segments_by_axis.iter().map(|s| s.len()).product();
        let bounding_boxes = if total_bounding_boxes <= 1000 {
            vec![BoundingBox(
                segments_by_axis
                    .iter()
                    .map(|_| 0..=std::usize::MAX)
                    .collect(),
            )]
        } else {
            segments_by_axis
                .iter()
                .multi_cartesian_product()
                .map(|segments_group| BoundingBox(segments_group.into_iter().cloned().collect()))
                .collect()
        };

        //
        // Find the patches we need to fill all the bounding boxes
        //

        // TODO: This could be async
        let mut patch_ids = HashSet::new();
        for bbox in &bounding_boxes {
            for patch_id in self.get_patches_by_bounding_box(&quilt_name, bbox)? {
                patch_ids.insert(patch_id);
            }
        }

        //
        // Download and apply all the patches
        //

        // TODO: This should definitely be async or at least concurrent
        let mut target_patch = Patch::from_axes(axes)?;
        for patch_id in patch_ids {
            let source_patch = self.get_patch(patch_id)?;
            target_patch.apply(&source_patch)?;
        }

        Ok(target_patch)
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

    fn create_quilt(&self, quilt_name: &str, axes_names: &[&str]) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT INTO quilt(quilt_name, axes) VALUES (?, ?);",
            &[&quilt_name, &serde_json::to_string(axes_names)?.as_ref()],
        )?;
        Ok(())
    }

    fn get_patches_by_bounding_box(
        &self,
        quilt_name: &str,
        bound: &BoundingBox,
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>> {
        // TODO: Verify that the dimensions match what we see in the quilt
        // The SQL formatting happens here because patch_tree
        let bound = (0..3)
            .map(|i| bound.0.get(i).cloned().unwrap_or(0..=1 << 30))
            .collect_vec();

        // Fetch patch ID's first, and then get them one by one. This is so we don't concurrently have multiple connections open.
        let mut patch_ids: Vec<PatchID> = vec![];
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
                    &(*bound[0].end() as i64) as &dyn ToSql,
                    &(*bound[0].start() as i64),
                    &(*bound[1].end() as i64),
                    &(*bound[1].start() as i64),
                    &(*bound[2].end() as i64),
                    &(*bound[2].start() as i64),
                    &quilt_name,
                ],
                |r| r.get(0),
            )?
        {
            patch_ids.push(patch_id?);
        }

        Ok(Box::new(patch_ids.into_iter()))
    }

    fn get_patch(&self, id: PatchID) -> Fallible<Patch<f32>> {
        let res: Vec<u8> = self.get_conn()?.query_row(
            "SELECT content FROM patch WHERE patch_id = ?",
            &[&id],
            |r| r.get(0),
        )?;
        Ok(bincode::deserialize(&res[..])?)
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
    fn union_axis(&self, new_axis: &Axis) -> Fallible<()> {
        let conn = self.get_conn()?;
        // TODO: determine if a race condition here is possible in an async environment
        conn.execute("BEGIN;", NO_PARAMS)?;

        let res: Option<Vec<u8>> = conn
            .query_row(
                "SELECT content FROM axis_content WHERE axis_name = ?",
                &[&new_axis.name],
                |r| r.get(0),
            )
            .optional()?;

        let mut existing_axis = match res {
            Some(x) => bincode::deserialize::<Axis>(&x[..])?.check_distinct()?,
            None => Axis::empty(&new_axis.name),
        };
        existing_axis.union(new_axis);

        conn.execute(
            "INSERT OR REPLACE INTO axis_content(axis_name, content) VALUES (?,?);",
            &[
                &new_axis.name as &dyn ToSql,
                &bincode::serialize(&existing_axis)?,
            ],
        )?;

        conn.execute("COMMIT;", NO_PARAMS)?;

        Ok(())
    }
}
