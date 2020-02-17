#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
use rocket::State;
use rocket_contrib::json::Json;
use std::collections::HashMap;
use std::sync::Arc;
use stoicheia::*;

//
// Catalog and Quilt metadata
//
#[get("/catalog")]
fn list_catalog(catalog: State<Arc<dyn Catalog>>) -> Fallible<Json<HashMap<String, QuiltMeta>>> {
    Ok(Json(catalog.list_quilts()?))
}

#[get("/quilt/<name>")]
fn get_quilt_meta(catalog: State<Arc<dyn Catalog>>, name: String) -> Fallible<Json<QuiltMeta>> {
    Ok(Json(catalog.get_quilt_meta(&name)?))
}

#[post("/catalog", format = "json", data = "<meta>")]
fn create_quilt(catalog: State<Arc<dyn Catalog>>, meta: Json<QuiltMeta>) -> Fallible<()> {
    Ok(catalog.create_quilt(meta.into_inner())?)
}

//
// Quilt patching
//

/// Get a slice out of a quilt
#[post("/quilt/slice/<quilt_name>", format = "json", data = "<patch_request>")]
fn get_patch(
    catalog: State<Arc<dyn Catalog>>,
    quilt_name: String,
    patch_request: Json<PatchRequest>,
) -> Fallible<Json<Patch<f32>>> {
    Ok(Json(
        catalog
            .fetch(&quilt_name, patch_request.into_inner())?,
    ))
}

/// Apply a patch to a quilt
#[patch("/quilt/slice/<quilt_name>", format = "json", data = "<pat>")]
fn apply_patch(
    catalog: State<Arc<dyn Catalog>>,
    quilt_name: String,
    pat: Json<Patch<f32>>,
) -> Fallible<()> {
    Ok(catalog.get_quilt(&quilt_name)?.apply(pat.into_inner())?)
}

/// Get a copy of a whole axis.
#[get("/axis/<name>")]
fn get_axis(catalog: State<Arc<dyn Catalog>>, name: String) -> Fallible<Json<Axis>> {
    Ok(Json(catalog.get_axis(&name)?))
}

/// Append some values to the end of an axis
///
/// This is a relatively cheap operation and doesn't require patches to be rebuilt.
#[patch("/axis/<name>", format = "json", data = "<labels>")]
fn append_axis(
    catalog: State<Arc<dyn Catalog>>,
    name: String,
    labels: Json<Vec<Label>>,
) -> Fallible<()> {
    Ok(catalog.union_axis(&Axis::new(name, labels.into_inner())?)?)
}

/// Create the rocket server separate from launching it, so we can test it
fn make_rocket(cat: Arc<dyn Catalog>) -> rocket::Rocket {
    rocket::ignite()
        .mount(
            "/",
            routes![
                list_catalog,
                get_quilt_meta,
                create_quilt,
                get_patch,
                apply_patch,
                get_axis,
                append_axis
            ],
        )
        .manage(cat)
}

fn main() {
    let catalog = SQLiteCatalog::connect("./stoicheia-storage.db".into()).unwrap();
    make_rocket(catalog).launch();
}

#[cfg(test)]
mod tests {
    use super::make_rocket;
    use rocket::http::ContentType;
    use rocket::http::Status;
    use rocket::local::Client;
    use stoicheia::*;

    #[test]
    fn quilt_axis_patch_test() {
        //
        // Create and retrieve quilts, axes, and patches
        //
        let catalog = SQLiteCatalog::connect_in_memory();
        let client = Client::new(make_rocket(catalog)).expect("valid rocket instance");

        {
            //
            // Create a quilt
            //
            let response = client
                .post("/catalog")
                .header(ContentType::JSON)
                .body(
                    r#"
                    {
                        "name": "sales",
                        "axes": ["item", "store", "day"]
                    }"#,
                )
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
        }

        {
            //
            // Retrieve a quilt
            //
            let mut response = client.get("/quilt/sales").dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(
                response.body_string(),
                Some(r#"{"name":"sales","axes":["item","store","day"]}"#.into())
            );
        }

        {
            //
            // Create axes
            //
            let response = client
                .patch("/axis/item")
                .header(ContentType::JSON)
                .body("[-4, 10]")
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            let response = client
                .patch("/axis/store")
                .header(ContentType::JSON)
                .body("[0, -12, 3]")
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            let response = client
                .patch("/axis/day")
                .header(ContentType::JSON)
                .body("[10, 11, 12, 14]")
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
        }

        {
            //
            // Retrieve an axis
            //
            let mut response = client.get("/axis/store").dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(
                response.body_string(),
                Some(r#"{"name":"store","labels":[0,-12,3]}"#.into())
            );
        }

        {
            //
            // Create a patch
            //

            // Store axis is not increasing but it does match the axis
            let patch_text = r#"
            {
                "axes": [
                    {
                        "name": "item",
                        "labels": [-4, 10]
                    },
                    {
                        "name": "store",
                        "labels": [0, -12, 3]
                    },
                    {
                        "name": "day",
                        "labels": [10, 11, 12, 14]
                    }
                ],
                "dense": {
                    "v": 1,
                    "dim": [2, 3, 4],
                    "data": [
                        0.01, 0.02, 0.03, 0.04,
                        0.05, 0.06, 0.07, 0.08,
                        0.09, 0.10, 0.11, 0.12,

                        0.01, 0.02, 0.03, 0.04,
                        0.05, 0.06, 0.07, 0.08,
                        0.09, 0.10, 0.11, 0.12
                    ]
                }   
            }"#;

            let response = client
                .patch("/quilt/slice/sales")
                .header(ContentType::JSON)
                .body(patch_text)
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
        }

        {
            //
            // Get one element from the patch
            //
            let mut response = client
                .post("/quilt/slice/sales")
                .header(ContentType::JSON)
                .body(
                    r#"[
                        {"type": "Labels", "name": "item", "labels": [-4]},
                        {"type": "Labels", "name": "store", "labels": [3]},
                        {"type": "Labels", "name": "day",  "labels": [12]}
                    ]"#,
                )
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(response.body_string(), Some(r#"{"axes":[{"name":"item","labels":[-4]},{"name":"store","labels":[3]},{"name":"day","labels":[12]}],"dense":{"v":1,"dim":[1,1,1],"data":[0.11]}}"#.into()));

            let mut response = client
                .post("/quilt/slice/sales")
                .header(ContentType::JSON)
                .body(
                    r#"[
                        {"type": "RangeInclusive", "name": "item", "start": -4, "end": -4},
                        {"type": "RangeInclusive", "name": "store", "start": 3, "end": 3},
                        {"type": "RangeInclusive", "name": "day", "start": 12, "end": 12}
                    ]"#,
                )
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(response.body_string(), Some(r#"{"axes":[{"name":"item","labels":[-4]},{"name":"store","labels":[3]},{"name":"day","labels":[12]}],"dense":{"v":1,"dim":[1,1,1],"data":[0.11]}}"#.into()));
        }

        {
            //
            // Retrieve the whole patch with transposition
            //
            let mut response = client
                .post("/quilt/slice/sales")
                .header(ContentType::JSON)
                .body(
                    r#"[
                        {"type": "All", "name": "item"},
                        {"type": "Labels", "name": "store", "labels": [-12, 0, 3]},
                        {"type": "All", "name": "day"}
                    ]"#,
                )
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(response.body_string(), Some(r#"{"axes":[{"name":"item","labels":[-4,10]},{"name":"store","labels":[-12,0,3]},{"name":"day","labels":[10,11,12,14]}],"dense":{"v":1,"dim":[2,3,4],"data":[0.05,0.06,0.07,0.08,0.01,0.02,0.03,0.04,0.09,0.1,0.11,0.12,0.05,0.06,0.07,0.08,0.01,0.02,0.03,0.04,0.09,0.1,0.11,0.12]}}"#.into()));
        }

        {
            //
            // Retrieve the whole patch
            //
            let mut response = client
                .post("/quilt/slice/sales")
                .header(ContentType::JSON)
                .body(
                    r#"[
                        {"type": "All", "name": "item"},
                        {"type": "All", "name": "store"},
                        {"type": "All", "name": "day"}
                    ]"#,
                )
                .dispatch();
            assert_eq!(response.status(), Status::Ok);
            assert_eq!(response.body_string(), Some(r#"{"axes":[{"name":"item","labels":[-4,10]},{"name":"store","labels":[0,-12,3]},{"name":"day","labels":[10,11,12,14]}],"dense":{"v":1,"dim":[2,3,4],"data":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12]}}"#.into()));
        }
    }
}
