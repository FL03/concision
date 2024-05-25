/*
    Appellation: entropy <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Entropy<T = Self> {
    type Output;

    fn cross_entropy(&self, target: &T) -> Self::Output;
}

pub struct CrossEntropy;
