use crate::Fallible;
use itertools::Itertools;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;
use thread_local::CachedThreadLocal;

use crate::{
    Axis, AxisSegment, AxisSelection, BoundingBox, Patch, PatchID, PatchRequest, Quilt, QuiltMeta,
    StoiError,
};

/// A connection to a tensor storage
pub trait Catalog: Send + Sync {
    /// Get a quilt by name
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt>;

    /// Get only the metadata associated with a quilt by name
    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta>;

    /// Create a new quilt, and create new axes for it if necessary
    fn create_quilt(&self, quilt_name: &str, axes_names: &[&str], ignore_if_exists: bool) -> Fallible<()>;

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

    /// Create an axis, possibly ignoring it if it exists
    fn create_axis(&self, name: &str, ignore_if_exists: bool) -> Fallible<()>;

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

/// An implementation of tensor storage on SQLite
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
            conn.busy_timeout(std::time::Duration::from_secs(5))?;
            conn.execute_batch("
                CREATE TABLE IF NOT EXISTS quilt(
                    quilt_name TEXT PRIMARY KEY COLLATE NOCASE,
                    axes TEXT NOT NULL CHECK (json_valid(axes))
                ) WITHOUT ROWID;

                -- Later see if an r-tree actually changes performance
                CREATE TABLE IF NOT EXISTS patch (
                    patch_id INTEGER PRIMARY KEY,
                    dim_0_min, dim_0_max,
                    dim_1_min, dim_1_max,
                    dim_2_min, dim_2_max,
                    dim_3_min, dim_3_max
                );

                CREATE TABLE IF NOT EXISTS patch_content(
                    patch_id INTEGER PRIMARY KEY,
                    content BLOB
                );

                CREATE TABLE IF NOT EXISTS axis_content(
                    axis_name TEXT PRIMARY KEY,
                    content BLOB
                );
            ")?;
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

    /// Create a quilt, and create axes as necessary to make it
    fn create_quilt(&self, quilt_name: &str, axes_names: &[&str], ignore_if_exists: bool) -> Fallible<()> {
        let conn = self.get_conn()?;
        for axis_name in axes_names {
            self.create_axis(axis_name, true)?;
        }
        conn.execute(
            &format!(
                "INSERT {} INTO quilt(quilt_name, axes) VALUES (?, ?);",
                if ignore_if_exists {"OR IGNORE"} else {""}
            ),
            &[&quilt_name, &serde_json::to_string(axes_names)?.as_ref()],
        )?;
        Ok(())
    }

    fn get_patches_by_bounding_box(
        &self,
        _quilt_name: &str,
        bound: &BoundingBox,
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>> {
        // TODO: Verify that the dimensions match what we see in the quilt
        // The SQL formatting happens here because patch_tree
        let bound = (0..4)
            .map(|i| bound.0.get(i).cloned().unwrap_or(0..=1 << 30))
            .collect_vec();

        // Fetch patch ID's first, and then get them one by one. This is so we don't concurrently have multiple connections open.
        let mut patch_ids: Vec<PatchID> = vec![];
        for patch_id in self
            .get_conn()?
            .prepare(
                "SELECT patch_id
            FROM patch
            WHERE
                dim_0_min <= ?
                AND dim_0_max >= ?
                AND dim_1_min <= ?
                AND dim_1_max >= ?
                AND dim_2_min <= ?
                AND dim_2_max >= ?
                AND dim_3_min <= ?
                AND dim_3_max >= ?
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
                    &(*bound[3].end() as i64),
                    &(*bound[3].start() as i64),
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
            "SELECT content FROM patch_content WHERE patch_id = ?",
            &[&id],
            |r| r.get(0),
        )?;
        Ok(bincode::deserialize(&res[..])?)
    }

    fn put_patch(&self, id: PatchID, pat: Patch<f32>) -> Fallible<()> {
        let conn = self.get_conn()?;
        conn.execute("BEGIN;", NO_PARAMS)?;
        conn.execute(
            "INSERT OR REPLACE INTO patch(
                patch_id,
                dim_0_min, dim_0_max,
                dim_1_min, dim_1_max,
                dim_2_min, dim_2_max,
                dim_3_min, dim_3_max
            ) VALUES (?,?,?,?,?,?,?,?,?);",
            &[
                &id as &dyn ToSql,
                // TODO: HACK: Support real bounding boxes
                &(-1 << 30), &(1 << 30),
                &(-1 << 30), &(1 << 30),
                &(-1 << 30), &(1 << 30),
                &(-1 << 30), &(1 << 30),
            ],
        )?;
        // TODO: If this serialize fails it will deadlock the connection by not rolling back
        conn.execute(
            "INSERT OR REPLACE INTO patch_content(patch_id, content) VALUES (?,?);",
            &[&id as &dyn ToSql, &bincode::serialize(&pat)?],
        )?;
        conn.execute("COMMIT;", NO_PARAMS)?;
        Ok(())
    }

    fn create_axis(&self, axis_name: &str, ignore_if_exists: bool) -> Fallible<()> {
        self.get_conn()?
            .execute(
                &format!(
                    "INSERT {} INTO axis_content(axis_name, content) VALUES (?,?);",
                    if ignore_if_exists { "OR IGNORE" } else { "" }
                ),
            &[
                &axis_name as &dyn ToSql,
                &bincode::serialize(&Axis::empty(axis_name))?,
            ])?;
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
        // But if we use a transaction it may deadlock - what to do??
        //conn.execute("BEGIN;", NO_PARAMS)?;

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
        
        // TODO: If this serialize fails it will deadlock the connection by not rolling back
        conn.execute(
            "INSERT OR REPLACE INTO axis_content(axis_name, content) VALUES (?,?);",
            &[
                &new_axis.name as &dyn ToSql,
                &bincode::serialize(&existing_axis)?,
            ],
        )?;

        //conn.execute("COMMIT;", NO_PARAMS)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{AxisSelection, Axis, Catalog, Label, SQLiteCatalog, Patch};


    #[test]
    fn test_create_axis() {
        let cat = SQLiteCatalog::connect_in_memory().unwrap();
        cat.create_axis("xjhdsa", false)
            .expect("Should be fine to create one that doesn't exist yet");
        cat.create_axis("xjhdsa", true)
            .expect("Should be fine to try to create an axis that exists");
        cat.create_axis("xjhdsa", false)
            .expect_err("Should fail to create duplicate axis");
        
        cat.get_axis("uyiuyoiuy")
            .expect_err("Should throw an error for an axis that doesn't exist.");
        let mut ax = cat.get_axis("xjhdsa")
            .expect("Should be able to get an axis I just made");
        assert_eq!(ax.labels(), &[]);

        ax = Axis::new("uyiuyoiuy", vec![Label(1), Label(5)])
            .expect("Should be able to create an axis");

        // Union an axis
        cat.union_axis(&ax)
            .expect("Should be able to union an axis");
        ax = cat.get_axis("uyiuyoiuy")
            .expect("Axis should exist after union");
        assert_eq!(ax.labels(), &[Label(1), Label(5)]);

        cat.union_axis(&ax)
            .expect("Union twice is a no-op");
        ax = cat.get_axis("uyiuyoiuy")
            .expect("Axis should still exist after second union");
        assert_eq!(ax.labels(), &[Label(1), Label(5)]);

        cat.union_axis(&Axis::new("uyiuyoiuy", vec![Label(0), Label(5)])
        .unwrap())
            .expect("Union should append");
        ax = cat.get_axis("uyiuyoiuy").unwrap();
        assert_eq!(ax.labels(), &[Label(1), Label(5), Label(0)]);
    }

    #[test]
    fn test_create_quilt() {
        let cat = SQLiteCatalog::connect_in_memory().unwrap();
        // This should automatically create the axes as well, so it doesn't complain
        cat.create_quilt("sales", &["itm", "lct", "day"], true).unwrap();
    }

    #[test]
    fn test_basic_fetch() {
        let cat = SQLiteCatalog::connect_in_memory().unwrap();
        cat.create_quilt("sales", &["itm", "lct", "day"], true).unwrap();

        // This should assume the axes' labels exist if you specify them, but not if you don't
        let mut pat = cat.fetch(
            "sales",
            vec![
                AxisSelection::All { name: "itm".into() },
                AxisSelection::Labels {
                    name: "itm".into(),
                    labels: vec![Label(1)],
                },
            ],
        )
        .unwrap();
        assert_eq!(pat.content().shape(), &[0, 1]);

        pat = Patch::from_axes(vec![
            Axis::range("itm", 9..12),
            Axis::range("xyz", 2..4),
        ]).unwrap();

        pat.content_mut().fill(1.0);
    }
}
