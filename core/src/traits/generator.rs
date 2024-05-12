/*
    Appellation: generator <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

/// This trait describes actors that can generate data
pub trait Generative<T> {
    type Output;

    fn generate(&self, args: T) -> Self::Output;
}
