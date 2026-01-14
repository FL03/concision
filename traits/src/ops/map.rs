/*
    Appellation: map <module>
    Created At: 2026.01.06:13:50:37
    Contrib: @FL03
*/
/// [`MapInto`] defines an interface for containers that can consume themselves to apply a given
/// function onto each of their elements.
pub trait MapInto<F, U>
where
    F: FnOnce(Self::Elem) -> U,
{
    type Cont<_T>;
    type Elem;

    fn mapi(self, f: F) -> Self::Cont<U>;
}

/// [`MapTo`] establishes an interface for containers capable of applying a given function onto
/// each of their elements, by reference.
pub trait MapTo<F, U>
where
    F: FnOnce(Self::Elem) -> U,
{
    type Cont<_T>;
    type Elem;

    fn mapt(&self, f: F) -> Self::Cont<U>;
}

/*
 ************* Implementations *************
*/
use ndarray::{Array, ArrayBase, Data, Dimension};

impl<U, V, F> MapInto<F, V> for Option<U>
where
    F: FnOnce(U) -> V,
{
    type Cont<W> = Option<W>;
    type Elem = U;

    fn mapi(self, f: F) -> Self::Cont<V> {
        self.map(|a| f(a))
    }
}

impl<'a, U, V, F> MapTo<F, V> for Option<&'a U>
where
    for<'b> F: FnOnce(&'b U) -> V,
{
    type Cont<W> = Option<W>;
    type Elem = &'a U;

    fn mapt(&self, f: F) -> Self::Cont<V> {
        self.as_ref().map(|a| f(a))
    }
}

impl<A, B, S, D, F> MapInto<F, B> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Cont<V> = Array<V, D>;
    type Elem = A;

    fn mapi(self, f: F) -> Self::Cont<B> {
        self.mapv(f)
    }
}

impl<'a, A, B, S, D, F> MapInto<F, B> for &'a ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Cont<V> = Array<V, D>;
    type Elem = A;

    fn mapi(self, f: F) -> Self::Cont<B> {
        self.mapv(f)
    }
}

impl<A, B, S, D, F> MapTo<F, B> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Cont<V> = Array<V, D>;
    type Elem = A;

    fn mapt(&self, f: F) -> Self::Cont<B> {
        self.mapv(f)
    }
}
