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
fn list_catalog(catalog: State<Arc<Catalog>>) -> Fallible<Json<HashMap<String, QuiltMeta>>> {
    Ok(Json(catalog.list_quilts()?))
}

#[get("/quilt/<name>/info")]
fn get_quilt_meta(catalog: State<Arc<Catalog>>, name: String) -> Fallible<Json<QuiltMeta>> {
    Ok(Json(catalog.get_quilt_meta(&name)?))
}

#[post("/catalog", format = "json", data = "<meta>")]
fn new_quilt(catalog: State<Arc<Catalog>>, meta: Json<QuiltMeta>) -> Fallible<()> {
    Ok(catalog.new_quilt(meta.into_inner())?)
}

//
// Quilt patching
//

#[patch("/quilt/<quilt_name>", format = "json", data = "<pat>")]
fn apply_patch(
    catalog: State<Arc<Catalog>>,
    quilt_name: String,
    pat: Json<Patch<f32>>,
) -> Fallible<()> {
    Ok(catalog.get_quilt(&quilt_name)?.apply(pat.into_inner())?)
}

#[post("/quilt/<quilt_name>", format = "json", data = "<patch_request>")]
fn get_patch(
    catalog: State<Arc<Catalog>>,
    quilt_name: String,
    patch_request: Json<PatchRequest>,
) -> Fallible<Json<Patch<f32>>> {
    Ok(Json(
        catalog
            .get_quilt(&quilt_name)?
            .assemble(patch_request.into_inner())?,
    ))
}


/// Create the rocket server separate from launching it, so we can test it
fn make_rocket() -> rocket::Rocket {
    let catalog = Catalog::connect("./stoicheia-storage.db".into()).unwrap();
    rocket::ignite()
        .mount("/", routes![list_catalog, get_quilt_meta, new_quilt])
        .manage(catalog)
}

fn main() {
    make_rocket().launch();
}

#[cfg(test)]
mod tests {
    use super::make_rocket;
    use rocket::local::Client;
    use rocket::http::Status;
    use rocket::http::ContentType;
    use stoicheia::*;

    #[test]
    fn create_new_quilt() {
        let client = Client::new(make_rocket()).expect("valid rocket instance");

        let response = client.post("/catalog")
            .header(ContentType::JSON)
            .body(r#"
                {
                    "name": "sales",
                    "axes": ["itm_nbr", "lct_nbr", "cal_dt"]
                }"#
            )
            .dispatch();
        assert_eq!(response.status(), Status::Ok);
        //assert_eq!(response.body_string(), Some("Hello, world!".into()));
    }
}