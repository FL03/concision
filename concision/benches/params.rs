/*
    appellation: params <benchmark>
    authors: @FL03
*/
extern crate concision as cnc;

use cnc::Params;
use cnc::init::NdRandom;
use core::hint::black_box;
use criterion::{BatchSize, BenchmarkId, Criterion};
use ndarray::Array1;

const SAMPLES: usize = 50;

const DEFAULT_DURATION_SECS: u64 = 10;

fn bench_params_forward(c: &mut Criterion) {
    // create a benchmark group for the Fibonacci iterator
    let mut group = c.benchmark_group("Params");
    // set the measurement time for the group
    group.measurement_time(std::time::Duration::from_secs(DEFAULT_DURATION_SECS));
    //set the sample size
    group.sample_size(SAMPLES);

    for &n in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("Params::forward", n), &n, |b, &x| {
            b.iter_batched(
                || {
                    let params = Params::<f64>::glorot_normal((n, 64));
                    let x = Array1::<f64>::linspace(0.0, 1.0, x);
                    (params, x)
                },
                |(params, x)| {
                    black_box(params.forward(&x));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion::criterion_group! {
    benches,
    bench_params_forward,
}

criterion::criterion_main! {
    benches
}
