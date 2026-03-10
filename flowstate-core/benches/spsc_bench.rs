//! Criterion benchmarks for the SPSC ring buffer.
//!
//! Measures single-threaded push/pop throughput and cross-thread throughput
//! to quantify the overhead of atomic operations and cache-line padding.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;

use flowstate_core::spsc::SpscRing;

/// Single-threaded: push N then pop N (no contention).
fn bench_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_single_thread");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let ring: SpscRing<u64> = SpscRing::new(n);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", n / 1000)),
            &n,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n as u64 {
                        ring.try_push(black_box(i)).unwrap();
                    }
                    for _ in 0..n {
                        black_box(ring.try_pop().unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

/// Cross-thread: producer pushes N, consumer pops N concurrently.
fn bench_cross_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_cross_thread");

    for n in [100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", n / 1000)),
            &n,
            |b, &n| {
                b.iter(|| {
                    let ring = Arc::new(SpscRing::new(4096));

                    let prod_ring = ring.clone();
                    let producer = thread::spawn(move || {
                        for i in 0..n as u64 {
                            prod_ring.push_spin(black_box(i));
                        }
                    });

                    let cons_ring = ring.clone();
                    let consumer = thread::spawn(move || {
                        for _ in 0..n {
                            black_box(cons_ring.pop_spin());
                        }
                    });

                    producer.join().unwrap();
                    consumer.join().unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_single_thread, bench_cross_thread);
criterion_main!(benches);
