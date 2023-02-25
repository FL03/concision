/*
    Appellation: complex <num>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/
use crate::Numerical;
use serde::{Deserialize, Serialize};
use std::{
    convert::From,
    ops::{Add, Mul},
};

// Define a trait for complex numbers.
pub trait Complex<T: Numerical> {
    fn build(re: T, im: T) -> Self;
    fn re(&self) -> T;
    fn im(&self) -> T;
}

// Implement the Complex trait for the tuple (f64, f64).
impl Complex<f64> for (f64, f64) {
    fn build(re: f64, im: f64) -> Self {
        (re, im)
    }

    fn re(&self) -> f64 {
        self.0
    }

    fn im(&self) -> f64 {
        self.1
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct C<T: Numerical = f64>(T, T);

impl<T: Numerical> C<T> {
    pub fn new(a: T, b: T) -> Self {
        Self(a, b)
    }
}

impl<T: Numerical> Complex<T> for C<T> {
    fn build(re: T, im: T) -> Self {
        Self(re, im)
    }
    fn re(&self) -> T {
        self.0
    }
    fn im(&self) -> T {
        self.1
    }
}

impl<T: Numerical> From<(T, T)> for C<T> {
    fn from(data: (T, T)) -> Self {
        Self::new(data.0, data.1)
    }
}

impl<T: Numerical> From<C<T>> for (T, T) {
    fn from(data: C<T>) -> (T, T) {
        (data.re(), data.im())
    }
}

// Implement the Add and Mul traits for complex numbers.
impl<T: Numerical> Add for C<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from((self.re() + other.re(), self.im() + other.im()))
    }
}

impl<T: Numerical> Mul for C<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::from((
            self.re() * other.re() - self.im() * other.im(),
            self.re() * other.im() + self.im() * other.re(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Define a holomorphic function that squares its input.
    fn square<T: Numerical>(z: C<T>) -> C<T> {
        z.clone() * z
    }

    #[test]
    fn test_square() {
        // Create a complex number (1 + i) and square it.
        let mut a = C::from((1.0, 1.0));
        a = square(a.clone());
        let b = C::new(0.0, 2.0);
        assert_eq!(&a, &b);
    }
}
