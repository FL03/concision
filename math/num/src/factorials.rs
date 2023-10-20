/*
    Appellation: factorials <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{Mul, Sub};
use std::str::FromStr;

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Factorial<T: Clone + FromStr + ToString + Mul<Output = T> + Sub<Output = T>>(T);

impl<T: Clone + FromStr + ToString + Mul<Output = T> + Sub<Output = T>> Factorial<T>
where
    <T as FromStr>::Err: std::fmt::Debug,
{
    pub fn new(data: T) -> Self {
        Self(Self::compute(data))
    }
    pub fn compute(data: T) -> T {
        let x = data.to_string();
        let b: T = "1".parse().ok().unwrap();
        match x.as_str() {
            "0" | "1" => b.clone(),
            _ => Self::compute(data.clone() - b) * data,
        }
    }
    pub fn data(&self) -> &T {
        &self.0
    }
    pub fn from_args(data: Vec<T>) -> Vec<Self> {
        data.iter()
            .map(|i| Self::new(i.clone()))
            .collect::<Vec<Self>>()
    }
}

pub fn factorial<T: Copy + num::Num + PartialEq + num::One + num::Zero>(data: T) -> T {
    if data.is_zero() || data.is_one() {
        return T::one();
    }
    factorial(data - T::one()) * data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(Factorial::new(0).data().clone(), 1);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4.0), 24.0);
        assert_eq!(factorial(5.0), 120.0);
    }
}
