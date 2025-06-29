/*
    appellation: loss <module>
    authors: @FL03
*/

/// The [`Loss`] trait defines a common interface for any custom loss function implementations.
/// This trait requires the implementor to define their algorithm for calculating the loss
/// between two values, `lhs` and `rhs`, which can be of different types, `X` and `Y`
/// respectively. These terms are used generically to allow for flexibility in the allowed
/// types, such as tensors, scalars, or other data structures while clearly defining the "order"
/// in which the operations are performed. It is most common to expect the `lhs` to be the
/// predicted output and the `rhs` to be the actual output, but this is not a strict requirement.
/// The trait also defines an associated type `Output`, which represents the type of the loss
/// value returned by the `loss` method. This allows for different loss functions to return
/// different types of loss values, such as scalars or tensors, depending on the specific
/// implementation of the loss function.
pub trait Loss<X, Y> {
    type Output;
    /// compute the loss between two values, `lhs` and `rhs`
    fn loss(&self, lhs: &X, rhs: &Y) -> Self::Output;
}
