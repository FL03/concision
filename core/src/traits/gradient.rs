/*
    appellation: gradient <module>
    authors: @FL03
*/

/// The [`Gradient`] trait defines a common interface for all gradients
pub trait Gradient<Rhs> {
    type Delta<_T>;
    type Output;

    fn grad(&self, rhs: &Rhs) -> Self::Delta<Self::Output>;
}
