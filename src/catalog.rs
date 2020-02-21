use crate::Fallible;
use itertools::Itertools;
use rusqlite::{ToSql, NO_PARAMS};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;
use thread_local::CachedThreadLocal;

use crate::{
    Axis, AxisSegment, AxisSelection, BoundingBox, Patch, PatchID, PatchRequest, Quilt,
    QuiltDetails, StoiError,
};

pub struct Catalog {
    storage: Arc<SQLiteCatalog>,
}
impl Catalog {
    /// Connect to a Stoicheia catalog.
    ///
    /// If url is "", then an in-memory catalog will be created.
    /// If url is a file path, then a new SQLite-based catalog will be created.
    /// Other URLs may be added to support other storage schemes.
    pub fn connect(url: &str) -> Fallible<Self> {
        Ok(if url == "" {
            Catalog {
                storage: SQLiteCatalog::connect_in_memory()?,
            }
        } else {
            Catalog {
                storage: SQLiteCatalog::connect(url.into())?,
            }
        })
    }

    /// List the quilts available in this Catalog
    pub fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>> {
        self.storage.list_quilts()
    }

    /// Create a quilt, and create axes as necessary to make it.
    ///
    /// With `ignore_if_exists=true`, you can make this idempotent rather than fail. In either case
    /// it will not replace or modify an existing quilt.
    pub fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()> {
        for axis_name in axes_names {
            self.storage.create_axis(axis_name, true)?;
        }
        self.storage
            .create_quilt(quilt_name, axes_names, ignore_if_exists)?;
        Ok(())
    }

    /// Get a quilt by name, as a convenient way to access patches
    ///
    /// This doesn't load any of the content into memory, it just provides a convenient
    /// way to access patches. As a result, this is pretty cheap, but it does incur IO
    /// to get the quilt's metadata so it is still fallible.
    pub fn get_quilt(&self, quilt_name: &str, tag: &str) -> Fallible<Quilt> {
        // This will fail if the quilt doesn't already exist
        self.storage.get_quilt_details(quilt_name)?;
        Ok(Quilt::new(quilt_name.into(), tag.into(), self))
    }

    /// Get details about a quilt by name
    ///
    /// What details are available may depend on the quilt, and fields are likely to
    /// be added with time (so be careful with serializing QuiltDetails)
    pub fn get_quilt_details(&self, quilt_name: &str) -> Fallible<QuiltDetails> {
        self.storage.get_quilt_details(quilt_name)
    }

    /// Create an empty axis
    pub fn create_axis(&self, axis_name: &str, ignore_if_exists: bool) -> Fallible<()> {
        self.storage.create_axis(axis_name, ignore_if_exists)
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    pub fn get_axis(&self, name: &str) -> Fallible<Axis> {
        self.storage.get_axis(name)
    }

    /// Replace the labels of an axis, in the order you would expect them to be stored.
    /// 
    /// Returns true iff the axis was mutated in the process 
    pub fn union_axis(&self, new_axis: &Axis) -> Fallible<bool> {
        // TODO: Race condition, needs a transaction
        let mut existing_axis = self
            .get_axis(&new_axis.name)
            .unwrap_or(Axis::empty(&new_axis.name));
        let mutated = existing_axis.union(new_axis);
        if mutated {
            self.storage.put_axis(&existing_axis)?;
        }
        Ok(mutated)
    }

    /// Fetch a patch from a quilt.
    ///
    /// - You can request any slice, and it will be assembled from the underlying commits.
    ///   - How many patches it's assembled from depends on the storage order
    ///     (which is the order the labels are specified in the axis, not in your request)
    /// - You can request elements you haven't initialized yet, and you'll get zeros.
    /// - You may only request patches up to 1 GB, as a safety valve
    pub fn fetch(
        &self,
        quilt_name: &str,
        tag: &str,
        mut request: PatchRequest,
    ) -> Fallible<Patch<f32>> {
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

        // They can't possibly use more than ten axes - just a safety measure.
        request.truncate(10);

        // Small note: we limit the axes here just as a safety measure
        for sel in request {
            let (axis, segments) = self.get_axis_from_selection(sel)?;
            axes.push(axis);
            segments_by_axis.push(segments);
        }

        // Tack on any axes they forgot
        for axis_name in self.get_quilt_details(quilt_name)?.axes {
            if axes.iter().find(|a| a.name == axis_name).is_none() {
                let (axis, segments) = self.get_axis_from_selection(AxisSelection::All{name: axis_name})?;
                axes.push(axis);
                segments_by_axis.push(segments);
            }
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
        for bbox in bounding_boxes {
            for patch_id in self
                .storage
                .get_patches_by_bounding_boxes(&quilt_name, &tag, &[bbox])?
            {
                patch_ids.insert(patch_id);
            }
        }

        //
        // Download and apply all the patches
        //

        // TODO: This should definitely be async or at least concurrent
        let mut target_patch = Patch::from_axes(axes)?;
        for patch_id in patch_ids {
            let source_patch = self.storage.get_patch(patch_id)?;
            target_patch.apply(&source_patch)?;
        }

        Ok(target_patch)
    }

    /// Use the actual axis values to resolve a request into specific labels
    ///
    /// This is necessary because we need to turn the axis labels into storage indices for range queries
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

    /// Commit a patch to a quilt.
    /// 
    /// Commits are a pretty expensive operation - the system is designed for more reads than writes.
    /// In specific, it will do at least all the following:
    /// 
    /// - Get quilt details, including full copies of all the axes
    /// - Check, compact, and compress all the patches, splitting and balancing search indices
    /// - Extend all the axes (if necessary) to include the area the patches cover
    /// - Upload all the patches and their data
    /// - Log the commit and change the tags to point to it
    /// 
    pub fn commit(
        &self,
        quilt_name: &str,
        parent_tag: Option<&str>,
        new_tag: Option<&str>,
        message: &str,
        patches: Vec<Patch<f32>>,
    ) -> Fallible<()> {
        // TODO: There are just so many race conditions here
        // TODO: Implement split/balance...
        // TODO: Think about axis versioning - maybe not a good idea anyway?

        // Check that the axes are consistent
        let quilt_details = self.get_quilt_details(quilt_name)?;
        for patch in &patches {
            if patch
                .axes()
                .iter()
                .map(|a| &a.name)
                .sorted()
                .ne(quilt_details.axes.iter().sorted())
            {
                return Err(StoiError::MisalignedAxes(format!(
                    "the quilt \"{}\" has axes [{}] but the patch has [{}], which doesn't match. Please note broadcasting is not (yet) supported. The patch axes should match exactly.",
                    quilt_name, quilt_details.axes.iter().join(", "), patch.axes().iter().map(|a|&a.name).join(", ")
                )));
            }
        }

        // Extend all axes as necessary to complete the patching
        for axis_name in &quilt_details.axes {
            let mut axis = self.get_axis(axis_name)?;
            let mut mutated = false;
            for patch in &patches {
                // Linear search over max 4 elements so don't sweat it
                mutated |= axis.union(&patch.axes().iter().find(|a| &a.name == axis_name).unwrap());
            }
            if mutated {
                // This is actually quite expensive so it's worth avoiding it where possible
                self.union_axis(&axis)?;
            }
        }

        self.storage.put_commit(
            quilt_name,
            parent_tag.unwrap_or("latest"),
            new_tag.unwrap_or("latest"),
            message,
            patches,
        )
    }
}

/// A connection to tensor storage
trait Storage: Send + Sync {
    /// Get only the metadata associated with a quilt by name
    fn get_quilt_details(&self, quilt_name: &str) -> Fallible<QuiltDetails>;

    /// Create a new quilt (doesn't create associated axes)
    fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()>;

    /// List all the quilts in the catalog
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>>;

    /// List all the patches that intersect a bounding box
    ///
    /// There may be false positives; some patches may not actually overlap
    /// There are no false negatives; all patches that overlap will be returned
    ///
    /// This method exists in case the database supports efficient multidimensional range queries
    /// such as SQLite or Postgres/PostGIS
    fn get_patches_by_bounding_boxes(
        &self,
        quilt_name: &str,
        tag: &str,
        bounds: &[BoundingBox],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>>;

    /// Get a single patch by ID
    fn get_patch(&self, id: PatchID) -> Fallible<Patch<f32>>;

    /// Create an axis, possibly ignoring it if it exists
    fn create_axis(&self, name: &str, ignore_if_exists: bool) -> Fallible<()>;

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis>;

    /// Overwrite a whole axis. Only do this through `Catalog.union_axis()`.
    fn put_axis(&self, axis: &Axis) -> Fallible<()>;

    /// Make changes to a tensor via a commit
    ///
    /// This is only available together, so that the underlying storage media can do this
    /// atomically without a complicated API
    fn put_commit(
        &self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: Vec<Patch<f32>>,
    ) -> Fallible<()>;
}

/// An implementation of tensor storage on SQLite
struct SQLiteCatalog {
    base: PathBuf,
    conn: CachedThreadLocal<rusqlite::Connection>,
}
impl SQLiteCatalog {
    /// Create a shared in-memory SQLite database
    pub fn connect_in_memory() -> Fallible<Arc<Self>> {
        let slf = Self {
            base: "file::memory:?cache=shared".into(),
            conn: CachedThreadLocal::new(),
        };
        slf.get_conn()?
            .execute_batch(include_str!("sqlite_catalog_schema.sql"))?;
        Ok(Arc::new(slf))
    }

    /// Connect to the underlying SQLite database
    pub fn connect(base: PathBuf) -> Fallible<Arc<Self>> {
        let slf = Self {
            base,
            conn: CachedThreadLocal::new(),
        };
        slf.get_conn()?
            .execute_batch(include_str!("sqlite_catalog_schema.sql"))?;
        Ok(Arc::new(slf))
    }

    /// Gets the thread-local SQLite connection
    ///
    /// There is a separate connection for each thread because it saves time connecting,
    /// but it can't be global because they can't be used from different threads simultaneously
    fn get_conn(&self) -> Fallible<&rusqlite::Connection> {
        self.conn.get_or_try(|| {
            let conn = rusqlite::Connection::open(&self.base)?;
            conn.busy_timeout(std::time::Duration::from_secs(5))?;
            Ok(conn)
        })
    }

    /// Put patch is only safe to do inside put_commit, so it's not part of Storage
    fn put_patch(
        &self,
        conn: &rusqlite::Connection,
        comm_id: i64,
        pat: Patch<f32>,
    ) -> Fallible<PatchID> {
        let patch_id = PatchID(self.gen_id());
        conn.execute("BEGIN;", NO_PARAMS)?;
        conn.execute(
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
                &(-1 << 30),
                &(1 << 30),
                &(-1 << 30),
                &(1 << 30),
                &(-1 << 30),
                &(1 << 30),
                &(-1 << 30),
                &(1 << 30),
            ],
        )?;
        // TODO: If this serialize fails it will deadlock the connection by not rolling back
        conn.execute(
            "INSERT OR REPLACE INTO PatchContent(patch_id, content) VALUES (?,?);",
            &[&patch_id as &dyn ToSql, &bincode::serialize(&pat)?],
        )?;
        conn.execute("COMMIT;", NO_PARAMS)?;
        Ok(patch_id)
    }

    /// Generate an id using the time plus a small salt
    fn gen_id(&self) -> i64 {
        chrono::Utc::now().timestamp_nanos() + rand::random::<i16>() as i64
    }
}

impl Storage for SQLiteCatalog {
    fn create_axis(&self, axis_name: &str, ignore_if_exists: bool) -> Fallible<()> {
        self.get_conn()?.execute(
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
        self.get_conn()?.execute(
            "INSERT OR REPLACE INTO AxisContent(axis_name, content) VALUES (?,?);",
            &[&axis.name as &dyn ToSql, &bincode::serialize(&axis)?],
        )?;
        Ok(())
    }

    /// Get all the labels of an axis, in the order you would expect them to be stored
    fn get_axis(&self, name: &str) -> Fallible<Axis> {
        let res: Vec<u8> = self.get_conn()?.query_row(
            "SELECT content FROM AxisContent WHERE axis_name = ?",
            &[&name],
            |r| r.get(0),
        )?;
        Ok(bincode::deserialize(&res[..])?)
    }

    /// Create a quilt
    fn create_quilt(
        &self,
        quilt_name: &str,
        axes_names: &[&str],
        ignore_if_exists: bool,
    ) -> Fallible<()> {
        self.get_conn()?.execute(
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
        Ok(self.get_conn()?.query_row_and_then(
            "SELECT quilt_name, axes FROM quilt WHERE quilt_name = ?",
            &[&quilt_name],
            |r| QuiltDetails::try_from(r),
        )?)
    }

    /// List the currently available quilts
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltDetails>> {
        let mut map = HashMap::new();
        for row in self
            .get_conn()?
            .prepare("SELECT quilt_name, axes FROM quilt;")?
            .query_map(NO_PARAMS, |r| QuiltDetails::try_from(r))?
        {
            let row = row?;
            map.insert(row.name.clone(), row);
        }
        Ok(map)
    }

    fn get_patches_by_bounding_boxes(
        &self,
        quilt_name: &str,
        tag: &str,
        bounding_boxes: &[BoundingBox],
    ) -> Fallible<Box<dyn Iterator<Item = PatchID>>> {
        // TODO: Verify that the dimensions match what we see in the quilt
        // Fetch patch ID's first, and then get them one by one. This is so we don't concurrently have multiple connections open.
        let mut patch_ids: Vec<PatchID> = vec![];
        for patch_id in self
            .get_conn()?
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
                    DISTINCT patch_id
                    FROM CommitAncestry
                    INNER JOIN Patch USING (comm_id)
                    --  INNER JOIN json_each(?) BoundingBox ON (
                    --          dim_0_min <= json_extract(value, '$[0]')
                    --      AND dim_0_max >= json_extract(value, '$[1]')
                    --      AND dim_1_min <= json_extract(value, '$[2]')
                    --      AND dim_1_max >= json_extract(value, '$[3]')
                    --      AND dim_2_min <= json_extract(value, '$[4]')
                    --      AND dim_2_max >= json_extract(value, '$[5]')
                    --      AND dim_3_min <= json_extract(value, '$[6]')
                    --      AND dim_3_max >= json_extract(value, '$[7]')
                    --  )
                    ORDER BY comm_id ASC
            ",
            )?
            .query_map(
                &[
                    &quilt_name as &dyn ToSql,
                    &tag,
                    // &serde_json::to_string(&bounding_boxes
                    //     .iter()
                    //     .map(|bx| [
                    //         bx.0.get(0).cloned().unwrap_or(0..=1<<30),
                    //         bx.0.get(1).cloned().unwrap_or(0..=1<<30),
                    //         bx.0.get(2).cloned().unwrap_or(0..=1<<30),
                    //         bx.0.get(3).cloned().unwrap_or(0..=1<<30),
                    //     ])
                    //     .map(|bx| [
                    //         *bx[0].start(),
                    //         *bx[0].end(),
                    //         *bx[1].start(),
                    //         *bx[1].end(),
                    //         *bx[2].start(),
                    //         *bx[2].end(),
                    //         *bx[3].start(),
                    //         *bx[3].end(),
                    //     ])
                    //     .collect_vec()
                    //)?
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
            "SELECT content FROM PatchContent WHERE patch_id = ?",
            &[&id],
            |r| r.get(0),
        )?;
        Ok(bincode::deserialize(&res[..])?)
    }

    // put_patch is part of Self, not Storage because you can only do it using put_commit()

    fn put_commit(
        &self,
        quilt_name: &str,
        parent_tag: &str,
        new_tag: &str,
        message: &str,
        patches: Vec<Patch<f32>>,
    ) -> Fallible<()> {
        let comm_id: i64 = self.gen_id();

        let conn = self.get_conn()?;
        for pat in patches {
            self.put_patch(conn, comm_id, pat)?;
        }
        // TODO: Race condition
        conn.execute(
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
        conn.execute(
            "INSERT OR REPLACE INTO Tag(
                quilt_name,
                tag_name,
                comm_id
            ) VALUES (?, ?, ?)",
            &[&quilt_name as &dyn ToSql, &new_tag, &comm_id],
        )?;
        Ok(())
    }
}

pub struct CommitDetails {
    comm_id: i64,
    quilt_name: String,
    tag: String,
    message: String,
    patches: Vec<PatchID>,
}

#[cfg(test)]
mod tests {
    use crate::{Axis, AxisSelection, Catalog, Label, Patch};

    #[test]
    fn test_create_axis() {
        let cat = Catalog::connect("").unwrap();
        cat.create_axis("xjhdsa", false)
            .expect("Should be fine to create one that doesn't exist yet");
        cat.create_axis("xjhdsa", true)
            .expect("Should be fine to try to create an axis that exists");
        cat.create_axis("xjhdsa", false)
            .expect_err("Should fail to create duplicate axis");

        cat.get_axis("uyiuyoiuy")
            .expect_err("Should throw an error for an axis that doesn't exist.");
        let mut ax = cat
            .get_axis("xjhdsa")
            .expect("Should be able to get an axis I just made");
        assert!(ax.labels() == &[] as &[Label]);

        ax = Axis::new("uyiuyoiuy", vec![1, 5]).expect("Should be able to create an axis");

        // Union an axis
        cat.union_axis(&ax)
            .expect("Should be able to union an axis");
        ax = cat
            .get_axis("uyiuyoiuy")
            .expect("Axis should exist after union");
        assert_eq!(ax.labels(), &[1, 5]);

        cat.union_axis(&ax).expect("Union twice is a no-op");
        ax = cat
            .get_axis("uyiuyoiuy")
            .expect("Axis should still exist after second union");
        assert_eq!(ax.labels(), &[1, 5]);

        cat.union_axis(&Axis::new("uyiuyoiuy", vec![0, 5]).unwrap())
            .expect("Union should append");
        ax = cat.get_axis("uyiuyoiuy").unwrap();
        assert_eq!(ax.labels(), &[1, 5, 0]);
    }

    #[test]
    fn test_create_quilt() {
        let cat = Catalog::connect("").unwrap();
        // This should automatically create the axes as well, so it doesn't complain
        cat.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();
    }

    #[test]
    fn test_basic_fetch() {
        let cat = Catalog::connect("").unwrap();
        cat.create_quilt("sales", &["itm", "lct", "day"], true)
            .unwrap();

        // This should assume the axes' labels exist if you specify them, but not if you don't
        let mut pat = cat
            .fetch(
                "sales",
                "latest",
                vec![
                    AxisSelection::All { name: "itm".into() },
                    AxisSelection::Labels {
                        name: "itm".into(),
                        labels: vec![1],
                    },
                ],
            )
            .unwrap();
        assert_eq!(pat.content().shape(), &[0, 1]);

        pat = Patch::from_axes(vec![Axis::range("itm", 9..12), Axis::range("xyz", 2..4)]).unwrap();

        pat.content_mut().fill(1.0);
    }
}
