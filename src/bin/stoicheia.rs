#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
use failure::Fallible;
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
fn put_quilt(catalog: State<Arc<dyn Catalog>>, meta: Json<QuiltMeta>) -> Fallible<()> {
    Ok(catalog.put_quilt(meta.into_inner())?)
}

//
// Quilt patching
//

#[post("/quilt/<quilt_name>", format = "json", data = "<patch_request>")]
fn get_patch(
    catalog: State<Arc<dyn Catalog>>,
    quilt_name: String,
    patch_request: Json<PatchRequest>,
) -> Fallible<Json<Patch<f32>>> {
    Ok(Json(
        catalog
            .get_quilt(&quilt_name)?
            .assemble(patch_request.into_inner())?,
    ))
}

#[patch("/quilt/<quilt_name>", format = "json", data = "<pat>")]
fn put_patch(
    catalog: State<Arc<dyn Catalog>>,
    quilt_name: String,
    pat: Json<Patch<f32>>,
) -> Fallible<()> {
    Ok(catalog.get_quilt(&quilt_name)?.apply(pat.into_inner())?)
}

/// Create the rocket server separate from launching it, so we can test it
fn make_rocket(cat: Arc<dyn Catalog>) -> rocket::Rocket {
    rocket::ignite()
        .mount(
            "/",
            routes![
                list_catalog,
                get_quilt_meta,
                put_quilt,
                get_patch,
                put_patch
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

    fn quilt_fixture() -> Client {
        let catalog = MemoryCatalog::new();
        let client = Client::new(make_rocket(catalog)).expect("valid rocket instance");

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
        std::mem::drop(response);

        client
    }

    #[test]
    fn create_new_quilt() {
        let client = quilt_fixture();

        let mut response = client.get("/quilt/sales").dispatch();
        assert_eq!(response.status(), Status::Ok);
        assert_eq!(
            response.body_string(),
            Some(r#"{"name":"sales","axes":["item","store","day"]}"#.into())
        );
    }

    #[test]
    fn get_and_set_patch() {
        let client = quilt_fixture();
        let patch_text = r#"
        {
            "axes": [
                {
                    "name": "item",
                    "labels": [-4, 10]
                },
                {
                    "name": "store",
                    "labels": [-12, 0, 3]
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
            .patch("/quilt/sales")
            .header(ContentType::JSON)
            .body(patch_text)
            .dispatch();
        assert_eq!(response.status(), Status::Ok);

        let mut response = client
            .post("/quilt/sales")
            .header(ContentType::JSON)
            .body(
                r#"{
                "axes": [
                    ["item", {"All": null}],
                    ["store", {"All": null}],
                    ["day", {"All": null}]
                ]
            }"#,
            )
            .dispatch();
        assert_eq!(response.status(), Status::Ok);
        assert_eq!(response.body_string(), Some(r#"{"axes":[{"name":"item","labels":[-4,10]},{"name":"store","labels":[-12,0,3]},{"name":"day","labels":[10,11,12,14]}],"dense":{"v":1,"dim":[2,3,4],"data":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12]}}"#.into()));
    }
}
