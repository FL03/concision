/*
    appellation: apply <module>
    authors: @FL03
*/
/// [`MapTo`] defines an interface for _functors_ capable of mapping a function over each
/// consituent element within the container.
pub trait MapTo<F, U>
where
    F: FnOnce(Self::Elem) -> U,
{
    type Cont<_T>;
    type Elem;

    fn apply(self, f: F) -> Self::Cont<U>;
}

/// The [`Apply`] establishes an interface for _owned_ containers that are capable of applying
/// some function onto their elements.
pub trait Apply<F, U>
where
    F: FnOnce(Self::Elem) -> U,
{
    type Cont<_T>;
    type Elem;

    fn apply(&self, f: F) -> Self::Cont<U>;
}
/// The [`ApplyOnce`] trait consumes the container and applies the given function to every
/// element before returning a new container with the results.
pub trait ApplyOnce<F, U>
where
    F: FnOnce(Self::Elem) -> U,
{
    type Cont<_T>;
    type Elem;

    fn apply_once(self, f: F) -> Self::Cont<U>;
}
/// [`ApplyMut`] provides an interface for mutable containers that can apply a function onto 
/// their elements, modifying them in place.
pub trait ApplyMut<T> {
    type Cont<_T>;

    fn apply_mut<'a, F>(&'a mut self, f: F)
    where
        T: 'a,
        F: FnMut(T) -> T;
}

/*
 ************* Implementations *************
*/
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};

impl<U, V, F> MapTo<F, V> for Option<U>
where
    F: FnOnce(U) -> V,
{
    type Cont<W> = Option<W>;
    type Elem = U;

    fn apply(self, f: F) -> Self::Cont<V> {
        self.map(|a| f(a))
    }
}

impl<A, B, S, D, F> Apply<F, B> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Cont<V> = Array<V, D>;
    type Elem = A;

    fn apply(&self, f: F) -> Self::Cont<B> {
        self.mapv(f)
    }
}

impl<A, S, D> ApplyMut<A> for ArrayBase<S, D, A>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply_mut<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}

impl<A, S, D> ApplyMut<A> for &mut ArrayBase<S, D, A>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply_mut<'b, F>(&'b mut self, f: F)
    where
        A: 'b,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
