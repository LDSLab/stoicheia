use failure::Fallible;
use rusqlite::{OptionalExtension, ToSql, NO_PARAMS};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;
use std::sync::Arc;
use thread_local::CachedThreadLocal;

use crate::{Patch, Quilt, QuiltMeta};

/// List of available tensors
pub struct Catalog {
    base: PathBuf,
    conn: CachedThreadLocal<rusqlite::Connection>,
}
impl Catalog {
    /// Connect to some persistence medium
    pub fn connect(base: PathBuf) -> Fallible<Arc<Self>> {
        Ok(Arc::new(Self {
            base,
            conn: CachedThreadLocal::new(),
        }))
    }

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

    pub fn new_quilt(&self, meta: QuiltMeta) -> Fallible<()> {
        self.get_conn()?
            .execute("INSERT INTO quilt(quilt_name, axes) VALUES (?, ?);", &[&meta.name, &serde_json::to_string(&meta.axes)?])?;
        Ok(())
    }

    /// List the currently available quilts
    pub fn list_quilts(&self) -> Fallible<HashMap<String, QuiltMeta>> {
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

    /// Get extended information about a quilt
    pub fn get_quilt_meta(&self, quilt_name: &str) -> Fallible<QuiltMeta> {
        Ok(self.get_conn()?.query_row_and_then(
            "SELECT quilt_name, axes FROM quilt WHERE quilt_name = ?",
            &[&quilt_name],
            |r| QuiltMeta::try_from(r),
        )?)
    }

    /// Open a quilt from this catalog
    pub fn get_quilt(&self, quilt_name: &str) -> Fallible<Quilt> {
        let meta = self.get_quilt_meta(quilt_name)?;
        Ok(Quilt::new(meta.name, &self))
    }

    pub fn put_patch(&self, quilt_name: &str, id: &str, pat: Patch<f32>) -> Fallible<()> {
        self.get_conn()?.execute(
            "INSERT INTO patch(quilt_name, id, content) VALUES (?,?,?);",
            &[&quilt_name as &dyn ToSql, &id, &bincode::serialize(&pat)?],
        )?;
        Ok(())
    }

    pub fn get_patch(&self, quilt_name: &str, id: &str) -> Fallible<Option<Patch<f32>>> {
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
}
