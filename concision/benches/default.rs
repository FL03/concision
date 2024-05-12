// bench.rs
#![feature(test)]

extern crate test;

use test::Bencher;

// bench: find the `BENCH_SIZE` first terms of the fibonacci sequence
const BENCH_SIZE: u32 = 20;

#[bench]
fn fibonacci(b: &mut Bencher) {
    // exact code to benchmark must be passed as a closure to the iter
    // method of Bencher
    b.iter(|| (0..BENCH_SIZE).map(fib::fibonacci).collect::<Vec<u128>>())
}

#[bench]
fn iter_fibonacci(b: &mut Bencher) {
    b.iter(|| fib::Fibonacci::new().take(BENCH_SIZE as usize).collect::<Vec<u32>>())
}

#[bench]
fn recursive_fibonacci(b: &mut Bencher) {
    // exact code to benchmark must be passed as a closure to the iter
    // method of Bencher
    b.iter(|| (0..BENCH_SIZE).map(fib::recursive_fibonacci).collect::<Vec<u128>>())
}

mod fib {
    /// fibonacci(n) returns the nth fibonacci number
    /// This function uses the definition of Fibonacci where:
    /// F(0) = F(1) = 1 and F(n+1) = F(n) + F(n-1) for n>0
    ///
    /// Warning: This will overflow the 128-bit unsigned integer at n=186
    pub fn fibonacci(n: u32) -> u128 {
        // Use a and b to store the previous two values in the sequence
        let mut a = 0;
        let mut b = 1;
        for _i in 0..n {
            // As we iterate through, move b's value into a and the new computed
            // value into b.
            let c = a + b;
            a = b;
            b = c;
        }
        b
    }

    /// fibonacci(n) returns the nth fibonacci number
    /// This function uses the definition of Fibonacci where:
    /// F(0) = F(1) = 1 and F(n+1) = F(n) + F(n-1) for n>0
    ///
    /// Warning: This will overflow the 128-bit unsigned integer at n=186
    pub fn recursive_fibonacci(n: u32) -> u128 {
        // Call the actual tail recursive implementation, with the extra
        // arguments set up.
        _recursive_fibonacci(n, 0, 1)
    }

    fn _recursive_fibonacci(n: u32, previous: u128, current: u128) -> u128 {
        if n == 0 {
            current
        } else {
            _recursive_fibonacci(n - 1, current, current + previous)
        }
    }
    
    pub struct Fibonacci {
        curr: u32,
        next: u32,
    }

    impl Fibonacci {
        pub fn new() -> Fibonacci {
            Fibonacci { curr: 0, next: 1 }
        }
    }

    impl Iterator for Fibonacci {
        type Item = u32;

        fn next(&mut self) -> Option<u32> {
            use core::mem::replace;
            let new_next = self.curr + self.next;
            let new_curr = replace(&mut self.next, new_next);

            Some(replace(&mut self.curr, new_curr))
        }
    }
}