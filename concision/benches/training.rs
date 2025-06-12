/*
    appellation: training <module>
    authors: @FL03
*/
extern crate concision as cnc;

use criterion::{BenchmarkId, Criterion};
use criterion::{criterion_group, criterion_main};

#[inline]
const fn do_something(_size: usize) {
    // Do something with the size
}

fn from_elem(c: &mut Criterion) {
    let size: usize = 1024;

    c.bench_with_input(BenchmarkId::new("input_example", size), &size, |b, &s| {
        b.iter(|| do_something(s));
    });
}

criterion_group!(benches, from_elem);
criterion_main!(benches);
