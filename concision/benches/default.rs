/*
    Appellation: default <module>
    Contrib: @FL03
*/
use core::hint::black_box;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use lazy_static::lazy_static;
use std::time::Duration;

const SAMPLES: usize = 50;
/// the default number of iterations to benchmark a method for
const N: usize = 20;
/// the default number of seconds a benchmark should complete in
const DEFAULT_DURATION_SECS: u64 = 10;

lazy_static! {
    /// a static reference to the duration of the benchmark
    static ref DURATION: Duration = Duration::from_secs(DEFAULT_DURATION_SECS);
}

fn bench_fib_func(c: &mut Criterion) {
    c.bench_function("fibonacci", |b| b.iter(|| fib::fibonacci(black_box(N))));
}

fn bench_fib_recursive(c: &mut Criterion) {
    c.bench_function("recursive_fibonacci", |b| {
        b.iter(|| fib::recursive_fibonacci(black_box(N)))
    });
}

fn bench_fib_iter(c: &mut Criterion) {
    let measure_for = Duration::from_secs(DEFAULT_DURATION_SECS);
    // create a benchmark group for the Fibonacci iterator
    let mut group = c.benchmark_group("Fibonacci Iter");
    // set the measurement time for the group
    group.measurement_time(measure_for);
    //set the sample size
    group.sample_size(SAMPLES);

    for &n in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("Fibonacci::compute", n), &n, |b, &x| {
            b.iter_batched(
                fib::Fibonacci::new,
                |mut fib| {
                    black_box(fib.compute(x));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}
// initialize the benchmark group
criterion_group! {
    benches,
    bench_fib_func,
    bench_fib_iter,
    bench_fib_recursive,
}
// This macro expands to a function named `benches`, which uses the given config
criterion_main!(benches);

pub mod fib {
    //! various implementations of the fibonacci sequence
    //!
    //! ##_Definition_:
    //!
    //! $F(0) = F(1) = 1 \text{ and } F(n+1) = F(n) + F(n-1) | \forall: n > 0$

    /// a simple implementation of the fibonacci sequence for benchmarking purposes
    /// **Warning:** This will overflow the 128-bit unsigned integer at n=186
    #[inline]
    pub fn fibonacci(n: usize) -> u128 {
        // Use a and b to store the previous two values in the sequence
        let mut a = 0;
        let mut b = 1;
        for _ in 0..n {
            // As we iterate through, move b's value into a and the new computed
            // value into b.
            let c = a + b;
            a = b;
            b = c;
        }
        b
    }
    /// a recursive implementation of the fibonacci sequence
    pub const fn recursive_fibonacci(n: usize) -> u128 {
        const fn _inner(n: usize, previous: u128, current: u128) -> u128 {
            if n == 0 {
                current
            } else {
                _inner(n - 1, current, current + previous)
            }
        }
        // Call the actual tail recursive implementation, with the extra
        // arguments set up.
        _inner(n, 0, 1)
    }
    /// A structural implementation of the fibonacci sequence that leverages the
    /// [`iter`](core::iter) as its backend
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Fibonacci {
        curr: u32,
        next: u32,
    }

    impl Fibonacci {
        /// returns a new instance of the fibonacci sequence, with `curr` set to 0 and `next`
        /// set to 1
        pub const fn new() -> Fibonacci {
            Fibonacci { curr: 0, next: 1 }
        }
        /// returns a copy of the current value
        pub const fn curr(&self) -> u32 {
            self.curr
        }
        /// returns a mutable reference to the current value
        pub const fn curr_mut(&mut self) -> &mut u32 {
            &mut self.curr
        }
        /// returns a copy of the next value
        pub const fn next(&self) -> u32 {
            self.next
        }
        /// returns a mutable reference to the next value
        pub const fn next_mut(&mut self) -> &mut u32 {
            &mut self.next
        }
        /// computes the nth value of the fibonacci sequence
        #[inline]
        pub fn compute(&mut self, n: usize) -> u32 {
            if let Some(res) = self.nth(n + 1) {
                return res;
            }
            panic!("Unable to compute the nth value of the fibonacci sequence...")
        }
        /// reset the instance to its default state, with `curr` set to 0 and `next` set to 1
        pub const fn reset(&mut self) -> &mut Self {
            self.set_curr(0).set_next(1)
        }
        /// compute the next value in the fibonacci sequence, using the current and next values
        #[inline]
        const fn compute_next(&self) -> u32 {
            self.curr() + self.next()
        }
        /// [`replace`](core::mem::replace) the current value with the given value, returning the
        /// previous value
        const fn replace_curr(&mut self, curr: u32) -> u32 {
            core::mem::replace(self.curr_mut(), curr)
        }
        /// [`replace`](core::mem::replace) the next value with the given value, returning the
        /// previous value
        const fn replace_next(&mut self, next: u32) -> u32 {
            core::mem::replace(self.next_mut(), next)
        }
        /// update the current value and return a mutable reference to the instance
        const fn set_curr(&mut self, curr: u32) -> &mut Self {
            self.curr = curr;
            self
        }
        /// update the next value and return a mutable reference to the instance
        const fn set_next(&mut self, next: u32) -> &mut Self {
            self.next = next;
            self
        }
        /// replace the next value with the given, using the previous next as the new current
        /// value, and returning the previous current value
        const fn update(&mut self, next: u32) -> u32 {
            let new = self.replace_next(next);
            self.replace_curr(new)
        }
    }

    impl Default for Fibonacci {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Iterator for Fibonacci {
        type Item = u32;

        fn next(&mut self) -> Option<u32> {
            // compute the new next value
            let new_next = self.compute_next();
            // replaces the current next with the new value and replaces the current value with
            // the previous next
            let prev = self.update(new_next);
            // return the previous current value
            Some(prev)
        }
    }
}
