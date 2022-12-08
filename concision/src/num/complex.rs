/*
    Appellation: complex <num>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/
use serde::{Deserialize, Serialize};
use std::{
    convert::From,
    ops::{Add, Mul},
};

// Define a trait for complex numbers.
pub trait Complex {
    fn new(re: f64, im: f64) -> Self;
    fn re(&self) -> f64;
    fn im(&self) -> f64;
}

// Implement the Complex trait for the tuple (f64, f64).
impl Complex for (f64, f64) {
    fn new(re: f64, im: f64) -> Self {
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
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl Complex for C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn re(&self) -> f64 {
        self.re
    }
    fn im(&self) -> f64 {
        self.im
    }
}

impl From<(f64, f64)> for C {
    fn from(data: (f64, f64)) -> Self {
        Self::new(data.0, data.1)
    }
}

impl From<C> for (f64, f64) {
    fn from(data: C) -> (f64, f64) {
        (data.re, data.im)
    }
}

// Implement the Add and Mul traits for complex numbers.
impl Add for C {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from((self.re + other.re, self.im + other.im))
    }
}

impl Mul for C {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::from((
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Define a holomorphic function that squares its input.
    fn square<T: Complex + Mul<Output = T> + Clone>(z: T) -> T {
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
