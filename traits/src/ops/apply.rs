/*
    appellation: apply <module>
    authors: @FL03
*/

/// [`Apply`] is a chainable, binary operator for applying some object onto the caller or their
/// elements.
pub trait Apply<Rhs> {
    type Output;

    fn apply(&self, rhs: Rhs) -> Self::Output;
}
/// The [`ApplyOnce`] trait consumes the container and applies the given function to every
/// element before returning a new container with the results.
pub trait ApplyOnce<Rhs> {
    type Output;

    fn apply_once(self, rhs: Rhs) -> Self::Output;
}
/// [`ApplyMut`] provides an interface for mutable containers that can apply a function onto
/// their elements, modifying them in place.
pub trait ApplyMut<Rhs> {
    fn apply_mut(&mut self, rhs: Rhs);
}

/*
 ************* Implementations *************
*/
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};

impl<U, V, F> ApplyOnce<F> for Option<U>
where
    F: FnOnce(U) -> V,
{
    type Output = Option<V>;

    fn apply_once(self, f: F) -> Self::Output {
        self.map(|a| f(a))
    }
}

impl<A, B, S, D, F> Apply<F> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Output = Array<B, D>;

    fn apply(&self, f: F) -> Self::Output {
        self.mapv(f)
    }
}

impl<A, S, D, F> ApplyMut<F> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: DataMut<Elem = A>,
    F: FnMut(A) -> A,
{
    fn apply_mut(&mut self, f: F) {
        self.mapv_inplace(f)
    }
}
