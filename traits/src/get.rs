/*
    appellation: get <module>
    authors: @FL03
*/

/// [`Get`] defines an interface for entities that can be accessed by a key; the design is
/// similar to the [`Index`](core::ops::Index) trait in the standard library, however, uses the
/// [`Borrow`](core::borrow::Borrow) trait to allow for more flexible key types.
pub trait Get<T> {
    type Key;
    /// returns a reference to the element at the specified index.
    fn get<Q>(&self, index: Q) -> Option<&T>
    where
        Q: core::borrow::Borrow<Self::Key>;
}
/// [`GetMut`] defines an interface for entities that can be accessed by a key; the design
/// is similar to the [`IndexMut`](core::ops::IndexMut) trait in the standard library
pub trait GetMut<T>: Get<T> {
    /// returns a mutable reference to the element at the specified index.
    fn get_mut<Q>(&mut self, index: Q) -> Option<&mut T>
    where
        Q: core::borrow::Borrow<Self::Key>;
}

/*
 ************* Implementations *************
*/

impl<X, Y> Get<Y> for &X
where
    X: Get<Y>,
{
    type Key = X::Key;

    fn get<Q>(&self, index: Q) -> Option<&Y>
    where
        Q: core::borrow::Borrow<Self::Key>,
    {
        (*self).get(index)
    }
}

impl<T> Get<T> for [T] {
    type Key = usize;

    fn get<Q>(&self, index: Q) -> Option<&T>
    where
        Q: core::borrow::Borrow<Self::Key>,
    {
        self.as_ref().get(*index.borrow())
    }
}
