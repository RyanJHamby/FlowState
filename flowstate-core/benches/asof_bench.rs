//! Criterion benchmarks for as-of join kernels.
//!
//! Run: cargo bench --manifest-path flowstate-core/Cargo.toml

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use flowstate_core::asof::parallel_scan::{
    par_merge_scan_backward, par_merge_scan_forward, par_merge_scan_nearest,
};
use flowstate_core::asof::scan::{merge_scan_backward, merge_scan_forward, merge_scan_nearest};

/// Generate sorted i64 timestamps with controlled density.
fn gen_sorted_ts(n: usize, seed: u64) -> Vec<i64> {
    let mut ts = Vec::with_capacity(n);
    let mut val: i64 = seed as i64;
    for i in 0..n {
        val += 1 + ((i as i64 * 7 + seed as i64) % 10);
        ts.push(val);
    }
    ts
}

fn bench_merge_scan_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_scan_backward");

    for size in [10_000, 100_000, 1_000_000] {
        let left = gen_sorted_ts(size, 42);
        let right = gen_sorted_ts(size / 2, 17);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(merge_scan_backward(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_merge_scan_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_scan_forward");

    for size in [10_000, 100_000, 1_000_000] {
        let left = gen_sorted_ts(size, 42);
        let right = gen_sorted_ts(size / 2, 17);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(merge_scan_forward(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_merge_scan_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_scan_nearest");

    for size in [10_000, 100_000, 1_000_000] {
        let left = gen_sorted_ts(size, 42);
        let right = gen_sorted_ts(size / 2, 17);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(merge_scan_nearest(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );
    }
    group.finish();
}

fn bench_with_tolerance(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_with_tolerance");

    let size = 1_000_000;
    let left = gen_sorted_ts(size, 42);
    let right = gen_sorted_ts(size / 2, 17);

    for tol in [100, 1_000, 10_000, i64::MAX] {
        let label = if tol == i64::MAX {
            "unlimited".to_string()
        } else {
            format!("{}ns", tol)
        };

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &tol,
            |b, &tol| {
                let tolerance = if tol == i64::MAX { None } else { Some(tol) };
                b.iter(|| {
                    black_box(merge_scan_backward(
                        black_box(&left),
                        black_box(&right),
                        tolerance,
                        true,
                    ))
                })
            },
        );
    }
    group.finish();
}

/// Compare sequential vs parallel scan at large scale.
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    for size in [100_000, 1_000_000, 5_000_000] {
        let left = gen_sorted_ts(size, 42);
        let right = gen_sorted_ts(size / 2, 17);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_backward", format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(merge_scan_backward(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel_backward", format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(par_merge_scan_backward(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sequential_nearest", format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(merge_scan_nearest(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel_nearest", format!("{}K", size / 1000)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(par_merge_scan_nearest(
                        black_box(&left),
                        black_box(&right),
                        None,
                        true,
                    ))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_merge_scan_backward,
    bench_merge_scan_forward,
    bench_merge_scan_nearest,
    bench_with_tolerance,
    bench_parallel_vs_sequential,
);
criterion_main!(benches);
