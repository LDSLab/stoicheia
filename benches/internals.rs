use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
        let mut target_patch = Patch::autogenerate(ContentPattern::Random, 1000);
        let source_patch = Patch::autogenerate(ContentPattern::Random, 1000);

        b.iter(|| target_patch.apply(black_box(&source_patch)))
    });

    {
        // A group of benchmarks that all benchmark serializing Patch,
        // because the compression and compaction can be slow
        let mut group = c.benchmark_group("Patch::serialize");

        // Two different ways to make the data
        for &pattern in &[
            ContentPattern::Empty,
            ContentPattern::Zero,
            ContentPattern::Random,
            ContentPattern::Sparse,
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
                        let patch = Patch::autogenerate(pattern, size as usize);
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
                        let patch_bytes = Patch::autogenerate(pattern, size as usize)
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
    for &pattern in &[ContentPattern::Sparse] {
        let mut cat = Catalog::connect("").unwrap();
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
                    &[&Patch::autogenerate(pattern, 1000)],
                ))
                .unwrap()
            })
        });

        let name = format!("Catalog::fetch 4MB read {:?} 100-patch commit", pattern);

        let ref quilt_name = format!("fetch_bench_quilt_{:?}", pattern);
        txn.create_quilt(quilt_name, &["dim0", "dim1"], true)
            .unwrap();
        let patches: Vec<_> = (0..100)
            .map(|_| Patch::autogenerate(pattern, 1000))
            .collect();
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
                black_box(
                    txn.fetch(
                        quilt_name,
                        "latest",
                        black_box(vec![
                            AxisSelection::StorageSlice(0, 1000),
                            AxisSelection::StorageSlice(0, 1000),
                        ]),
                    )
                    .unwrap(),
                )
            })
        });
    }
}

criterion_group!(benches, bench_axis, bench_patch, bench_commit);
criterion_main!(benches);
