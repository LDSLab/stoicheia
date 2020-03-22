use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand;
use stoicheia::*;
#[macro_use] extern crate lazy_static;

lazy_static! {
    static ref EMPTY_4MB_PATCH: Patch = Patch::build()
        .axis_range("dim0", black_box(1000..2000))
        .axis_range("dim1", black_box(0..1000))
        .content(None)
        .unwrap();
}

#[inline]
fn new_axis(labels: &[i64]) -> Axis {
    Axis::new("item", labels.to_owned()).unwrap()
}

pub fn bench_axis(c: &mut Criterion) {
    let labels: Vec<i64> = (0..100000).collect();
    c.bench_function("Axis::new ordered clone", |b| {
        b.iter(|| new_axis(black_box(&labels)))
    });
    let labels: Vec<i64> = (0..100000).map(|_| rand::random()).collect();
    c.bench_function("Axis::new random clone", |b| {
        b.iter(|| new_axis(black_box(&labels)))
    });
}

pub fn bench_patch(c: &mut Criterion) {
    c.bench_function("Patch::try_from_axes 2d clone", |b| {
        b.iter(|| {
            Patch::build()
                .axis_range("dim0", black_box(1000..2000))
                .axis_range("dim1", black_box(0..1000))
                .content(None)
                .unwrap();
        })
    });

    c.bench_function("Patch::apply perfect no-clone", |b| {
        let mut target_patch = Patch::build()
            .axis_range("dim0", 1000..2000)
            .axis_range("dim1", 0..1000)
            .content(None)
            .unwrap();

        b.iter(|| target_patch.apply(black_box(&EMPTY_4MB_PATCH)))
    });

    c.bench_function("Patch::apply subset no-clone", |b| {
        let mut target_patch = Patch::build()
            .axis_range("dim0", 1000..2000)
            .axis_range("dim1", 0..1000)
            .content(None)
            .unwrap();
        let source_patch = Patch::build()
            .axis_range("dim0", 1500..2000)
            .axis_range("dim1", 0..1000)
            .content(None)
            .unwrap();

        b.iter(|| target_patch.apply(black_box(&source_patch)))
    });

    {
        // A group of benchmarks that all benchmark serializing Patch,
        // because the compression and compaction can be slow
        let mut group = c.benchmark_group("Patch::serialize");

        let empty_content = |_| None;
        let random_content = |size| Some(ndarray::ArrayD::from_shape_fn(
            vec![ size as usize, size as usize ],
            |_| rand::random::<f32>()
        ));
        // Two different ways to make the data
        for (content_name, content_factory) in &[
            ("empty", &empty_content as &dyn Fn(usize) -> _),
            ("random", &random_content)] {

            // Three different sizes
            for &size in &[250, 500, 1000] {
                // Serialize
                group.throughput(criterion::Throughput::Bytes(size*size*4));
                group.sample_size(10).bench_function(format!("Patch::serialize {} {}x{}", content_name, size, size), |b| {
                    let patch = Patch::build()
                        .axis_range("dim0", 1000..1000+size as i64)
                        .axis_range("dim1", 0..size as i64)
                        .content(content_factory(size as usize))
                        .unwrap();
    
                    b.iter(|| black_box(&patch).serialize(None).unwrap() )
                });

                // Deserialize
                group.sample_size(10).bench_function(format!("Patch::deserialize {} {}x{}", content_name, size, size), |b| {
                    let patch_bytes = Patch::build()
                        .axis_range("dim0", 1000..1000+size as i64)
                        .axis_range("dim1", 0..size as i64)
                        .content(content_factory(size as usize))
                        .unwrap()
                        .serialize(None)
                        .unwrap();
    
                    b.iter(|| Patch::deserialize_from(black_box(&patch_bytes[..])).unwrap() )
                });
            }
        }
    }

    c.bench_function("Patch::apply overlap no-clone", |b| {
        let mut target_patch = Patch::build()
            .axis_range("dim0", 1500..2500)
            .axis_range("dim1", 0..1000)
            .content(None)
            .unwrap();

        b.iter(|| target_patch.apply(black_box(&EMPTY_4MB_PATCH)))
    });
}

pub fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("Catalog::commit");

    for &rewrites in &[1, 4, 16] {
        let name = format!("Catalog::commit 4MB empty total rewrite repeat-{}", rewrites);
        group.sample_size(10).bench_function(name, |b| {
            let catalog = Catalog::connect("").unwrap();
            catalog.create_quilt("quilt", &["dim0", "dim1"], true).unwrap();
            let patch = Patch::build()
                .axis_range("dim0", 1500..2500)
                .axis_range("dim1", 0..1000)
                .content(None)
                .unwrap();
            b.iter(|| {
                for _ in 0..rewrites {
                    catalog.commit("quilt", "latest", "latest", "message", vec![black_box(&patch)]).unwrap()
                }
            })
        });

        let name = format!("Catalog::commit 4MB read empty total rewrite {}-patch commit", rewrites);
        group.sample_size(10).bench_function(name, |b| {
            let catalog = Catalog::connect("").unwrap();
            catalog.create_quilt("quilt", &["dim0", "dim1"], true).unwrap();
            let patch = Patch::build()
                .axis_range("dim0", 1500..2500)
                .axis_range("dim1", 0..1000)
                .content(None)
                .unwrap();
            for _ in 0..rewrites {
                catalog.commit("quilt", "latest", "latest", "message", vec![black_box(&patch)]).unwrap()
            }
            b.iter(|| {
                catalog.fetch("quilt", "latest", vec![
                    AxisSelection::RangeInclusive{
                        name: "dim0".into(),
                        start: 1500,
                        end: 2499
                    },
                    AxisSelection::RangeInclusive{
                        name: "dim1".into(),
                        start: 0,
                        end: 999
                    }
                ]).unwrap()
            })
        });
    }
    
}
criterion_group!(benches, bench_axis, bench_patch, bench_commit);
criterion_main!(benches);
