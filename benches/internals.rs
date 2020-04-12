use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng; // This RNG is much faster and not secure but we don't need that
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use stoicheia::*;
#[macro_use]
extern crate lazy_static;
lazy_static! {
    static ref EMPTY_4MB_PATCH: Patch = Patch::build()
        .axis_range("dim0", black_box(1000..2000))
        .axis_range("dim1", black_box(0..1000))
        .content(None)
        .unwrap();
}

fn new_axis(labels: &[i64]) -> Axis {
    Axis::new("item", labels.to_owned()).unwrap()
}

#[derive(Debug, Clone, Copy)]
enum Pattern {
    Empty,
    Zero,
    Random,
    Sparse,
}
/// Create a new patch with some pattern of content
///
/// Almost everything is sensitive to the content, for example:
///     - Patch serialization is compressed, where empty is fast and random is slow
///     - Patch apply speed depends on how predictable memory access is
///     - Sparse areas can trigger compacting, saving space with expense of CPU
fn create_content(pattern: Pattern, size: usize) -> Patch {
    use Pattern::*;
    // This rng makes it easier to get a uniform range
    let mut rng = SmallRng::from_entropy();

    let content = match pattern {
        Empty => None,
        Zero => Some(ndarray::Array2::zeros([size, size]).into_dyn()),
        Random => Some(ndarray::ArrayD::from_shape_fn(
            vec![size as usize, size as usize],
            |_| rand::random::<f32>(),
        )),
        Sparse => {
            let mut canvas = ndarray::Array2::from_elem([size, size], std::f32::NAN);
            let nonzero_count = size * size / 1000;
            for _ in 0..nonzero_count {
                // This is indicative of the sparsity but not really the values
                let x = rng.gen_range(0, size);
                let y = rng.gen_range(0, size);
                canvas[[x, y]] = rng.gen_range(0f32, 5.0);
            }
            Some(canvas.into_dyn())
        }
    };

    // Let one axis be wildly sparse so every patch overlaps practically every other
    let dim0: Vec<i64> = (0..size * 5 / 4)
        .map(|_| rng.gen_range(-1000000, 1000000))
        .collect::<HashSet<_>>()
        .into_iter()
        .take(size)
        .collect();
    // Let the other be less sparse and clumpy
    let dim1_start = rng.gen_range(0i64, 25000);
    let dim1: Vec<i64> = (0..3 * size as i64)
        .step_by(3)
        .map(|lab| lab + dim1_start)
        .collect();

    Patch::build()
        .axis("dim0", &dim0)
        .axis("dim1", &dim1)
        .content(content)
        .unwrap()
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

    c.bench_function("Patch::apply overlap no-clone", |b| {
        let mut target_patch = create_content(Pattern::Random, 1000);
        let source_patch = create_content(Pattern::Random, 1000);

        b.iter(|| target_patch.apply(black_box(&source_patch)))
    });

    {
        // A group of benchmarks that all benchmark serializing Patch,
        // because the compression and compaction can be slow
        let mut group = c.benchmark_group("Patch::serialize");

        // Two different ways to make the data
        for &pattern in &[
            Pattern::Empty,
            Pattern::Zero,
            Pattern::Random,
            Pattern::Sparse,
        ] {
            for &compression in &[
                PatchCompressionType::Off,
                PatchCompressionType::LZ4 { quality: 1 },
            ] {
                let size = 1000;
                // Serialize
                group.throughput(criterion::Throughput::Bytes(size * size * 4));
                group.sample_size(10).bench_function(
                    format!(
                        "Patch::serialize {:?} {:?} {}x{}",
                        pattern, compression, size, size
                    ),
                    |b| {
                        let patch = create_content(pattern, size as usize);
                        b.iter(|| black_box(&patch).serialize(Some(compression)).unwrap())
                    },
                );

                // Deserialize
                group.sample_size(10).bench_function(
                    format!(
                        "Patch::deserialize {:?} {:?} {}x{}",
                        pattern, compression, size, size
                    ),
                    |b| {
                        let patch_bytes = create_content(pattern, size as usize)
                            .serialize(Some(compression))
                            .unwrap();

                        b.iter(|| Patch::deserialize_from(black_box(&patch_bytes[..])).unwrap())
                    },
                );
            }
        }
    }
}

pub fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("Catalog::commit");
    // Two different ways to make the data
    for &pattern in &[Pattern::Sparse] {
        let cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();
        let name = format!("Catalog::commit 4MB {:?} total rewrite", pattern);
        group.sample_size(10).bench_function(name, |b| {
            let ref quilt_name = format!("commit_bench_quilt_{:?}", pattern);
            txn.create_quilt(quilt_name, &["dim0", "dim1"], true)
                .unwrap();
            b.iter(|| {
                black_box(txn.create_commit(
                    quilt_name,
                    "latest",
                    "latest",
                    "message",
                    &[&create_content(pattern, 1000)],
                ))
                .unwrap()
            })
        });

        let name = format!("Catalog::fetch 4MB read {:?} 100-patch commit", pattern);

        let ref quilt_name = format!("fetch_bench_quilt_{:?}", pattern);
        txn.create_quilt(quilt_name, &["dim0", "dim1"], true)
            .unwrap();
        let patches: Vec<_> = (0..100).map(|_| create_content(pattern, 1000)).collect();
        txn.create_commit(
            quilt_name,
            "latest",
            "latest",
            "message",
            &patches.iter().collect::<Vec<_>>()[..],
        )
        .unwrap();

        group.sample_size(10).bench_function(name, |b| {
            b.iter(|| {
                black_box(txn.fetch(
                    quilt_name,
                    "latest",
                    black_box(vec![
                        AxisSelection::StorageSlice(0, 1000),
                        AxisSelection::StorageSlice(0, 1000),
                    ]),
                )
                .unwrap())
            })
        });
    }
}

criterion_group!(benches, bench_axis, bench_patch, bench_commit);
criterion_main!(benches);
