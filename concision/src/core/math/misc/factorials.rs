/*
    Appellation: factorials <module>
    Contributors: FL03 <jo3mccain@icloud.com> (https://gitlab.com/FL03)
    Description:
        ... Summary ...
*/
use std::{
    ops::{Mul, Sub},
    str::FromStr,
    string::ToString,
};

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Factorial<T: Clone + FromStr + ToString + Mul<Output=T> + Sub<Output=T>>(pub T);

impl<T: Clone + FromStr + ToString + Mul<Output=T> + Sub<Output=T>> Factorial<T>
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
    pub fn from_args(data: Vec<T>) -> Vec<Self> {
        data.iter()
            .map(|i| Self::new(i.clone()))
            .collect::<Vec<Self>>()
    }
}

pub fn factorial(data: usize) -> usize {
    match data {
        0 | 1 => 1,
        _ => factorial(data - 1) * data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(Factorial::new(0).0, 1)
    }
}
