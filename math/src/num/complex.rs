/*
    Appellation: complex <num>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Numerical;
use serde::{Deserialize, Serialize};
use std::ops;

// Define a trait for complex numbers.
pub trait Complex<T: Numerical> {
    fn new(re: T, im: T) -> Self;
    fn re(&self) -> T;
    fn im(&self) -> T;
}

// Implement the Complex trait for the tuple (f64, f64).
impl<T: Numerical> Complex<T> for (T, T) {
    fn new(re: T, im: T) -> Self {
        (re, im)
    }

    fn re(&self) -> T {
        self.0
    }

    fn im(&self) -> T {
        self.1
    }
}

impl<T: Numerical> Complex<T> for [T; 2] {
    fn new(re: T, im: T) -> Self {
        [re, im]
    }

    fn re(&self) -> T {
        self[0]
    }

    fn im(&self) -> T {
        self[1]
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct C<T: Numerical = f64>(T, T);

impl<T: Numerical> Complex<T> for C<T> {
    fn new(re: T, im: T) -> Self {
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
impl<T: Numerical> ops::Add for C<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from((self.re() + other.re(), self.im() + other.im()))
    }
}

impl<T> ops::AddAssign for C<T>
where
    T: Numerical + ops::AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<T: Numerical> ops::Div for C<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.re() * other.re() + other.im() * other.im();
        Self::from((
            (self.re() * other.re() + self.im() * other.im()) / denom,
            (self.im() * other.re() - self.re() * other.im()) / denom,
        ))
    }
}

impl<T: Numerical> ops::DivAssign for C<T> {
    fn div_assign(&mut self, other: Self) {
        let denom = other.re() * other.re() + other.im() * other.im();
        let re = (self.re() * other.re() + self.im() * other.im()) / denom;
        let im = (self.im() * other.re() - self.re() * other.im()) / denom;
        self.0 = re;
        self.1 = im;
    }
}

impl<T: Numerical> ops::Mul for C<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::from((
            self.re() * other.re() - self.im() * other.im(),
            self.re() * other.im() + self.im() * other.re(),
        ))
    }
}

impl<T: Numerical> ops::MulAssign for C<T> {
    fn mul_assign(&mut self, other: Self) {
        let re = self.re() * other.re() - self.im() * other.im();
        let im = self.re() * other.im() + self.im() * other.re();
        self.0 = re;
        self.1 = im;
    }
}

impl<T: Numerical> ops::Sub for C<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from((self.re() - other.re(), self.im() - other.im()))
    }
}

impl<T> ops::SubAssign for C<T>
where
    T: Numerical + ops::SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
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
