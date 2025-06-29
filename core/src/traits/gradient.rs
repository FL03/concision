/*
    appellation: gradient <module>
    authors: @FL03
*/

/// the [`Gradient`] trait defines the gradient of a function, which is a function that
/// takes an input and returns a delta, which is the change in the output with respect to
/// the input.
pub trait Gradient<T, D> {
    type Repr<_A>;
    type Delta<_S, _D>;

    fn grad(&self, rhs: &Self::Delta<Self::Repr<T>, D>) -> Self::Delta<Self::Repr<T>, D>;
}
