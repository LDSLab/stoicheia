use failure::Fallible;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::{Mutex, Arc};
use thread_local::CachedThreadLocal;

use crate::{Patch, Quilt, QuiltMeta};
pub trait Catalog: Send + Sync {
    fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt>;
    fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta>;
    fn put_quilt(&self, meta: QuiltMeta) -> Fallible<()>;
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>>;
    fn get_patch(&self, quilt_name: &str, id: &str) -> Fallible<Option<Patch<f32>>>;
    fn put_patch(&self, quilt_name: &str, id: &str, pat: Patch<f32>) -> Fallible<()>;
}

/// An in-memory catalog, meant for testing and dummy databases
pub struct MemoryCatalog {
    quilts: Mutex<HashMap<String, QuiltMeta>>,
    patches: Mutex<HashMap<String, Patch<f32>>>
}
impl MemoryCatalog {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            quilts: Mutex::from(HashMap::new()),
            patches: Mutex::from(HashMap::new()),
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
        self.quilts.lock()
            .expect("Memory catalog is corrupted.")
            .get(quilt_name)
            .ok_or(format_err!("No such quilt {}", quilt_name))
            .cloned()
    }
    fn put_quilt(&self, meta: QuiltMeta) -> Fallible<()> {
        self.quilts.lock()
            .expect("Memory catalog is corrupted.")
            .insert(meta.name.clone(), meta);
        Ok(())
    }
    fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>> {
        Ok(self.quilts.lock()
            .expect("Memory catalog is corrupted")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
    fn get_patch(&self, quilt_name: &str, id: &str) -> Fallible<Option<Patch<f32>>> {
        Ok(self.patches.lock()
            .expect("Memory catalog is corrupted")
            .get(&format!("{} {}", quilt_name, id))
            .cloned()
        )
    }
    fn put_patch(&self, quilt_name: &str, id: &str, pat: Patch<f32>) -> Fallible<()> {
        self.patches.lock()
            .expect("Memory catalog is corrupted")
            .insert(
                format!("{} {}", quilt_name, id),
                pat);
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
                "
                CREATE TABLE IF NOT EXISTS patch(
                    quilt_name TEXT NOT NULL COLLATE NOCASE,
                    id TEXT NOT NULL COLLATE NOCASE,
                    content BLOB,
                    PRIMARY KEY (quilt_name, id)
                ) WITHOUT ROWID;
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
        self.get_conn()?
            .execute("INSERT INTO quilt(quilt_name, axes) VALUES (?, ?);", &[&meta.name, &serde_json::to_string(&meta.axes)?])?;
        Ok(())
    }

    fn get_patch(&self, quilt_name: &str, id: &str) -> Fallible<Option<Patch<f32>>> {
        let res: Option<Vec<u8>> = self
            .get_conn()?
            .query_row(
                "SELECT content FROM patch WHERE quilt_name = ? AND id = ?",
                &[&quilt_name, &id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match res {
            None => None,
            Some(x) => bincode::deserialize(&x[..])?,
        })
    }

    fn put_patch(&self, quilt_name: &str, id: &str, pat: Patch<f32>) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT INTO patch(quilt_name, id, content) VALUES (?,?,?);",
            &[&quilt_name as &dyn ToSql, &id, &bincode::serialize(&pat)?],
        )?;
        Ok(())
    }
}