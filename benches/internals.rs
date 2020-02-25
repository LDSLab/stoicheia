use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand;
use stoicheia::*;

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
        let source_patch = target_patch.clone();

        b.iter(|| target_patch.apply(black_box(&source_patch)))
    });
}

criterion_group!(benches, bench_axis, bench_patch);
criterion_main!(benches);
