/*
    appellation: get <module>
    authors: @FL03
*/

/// [`Get`] defines an interface for entities that can be accessed by a key; the design is
/// similar to the [`Index`](core::ops::Index) trait in the standard library, however, uses the
/// [`Borrow`](core::borrow::Borrow) trait to allow for more flexible key types.
pub trait Get<Q> {
    type Key: ?Sized;
    type Output: ?Sized;
    /// returns a reference to the element at the specified index.
    fn get(&self, index: Q) -> Option<&Self::Output>
    where
        Self::Key: core::borrow::Borrow<Q>;
}
/// [`GetMut`] defines an interface for entities that can be accessed by a key; the design
/// is similar to the [`IndexMut`](core::ops::IndexMut) trait in the standard library
pub trait GetMut<T>: Get<T> {
    /// returns a mutable reference to the element at the specified index.
    fn get_mut<Q>(&mut self, index: Q) -> Option<&mut T>
    where
        Self::Key: core::borrow::Borrow<Q>;
}

/*
 ************* Implementations *************
*/

impl<Q, K, U, Y> Get<Q> for &U
where
    U: Get<Q, Key = K, Output = Y>,
{
    type Key = U::Key;
    type Output = Y;

    fn get(&self, index: Q) -> Option<&Self::Output>
    where
        Self::Key: core::borrow::Borrow<Q>,
    {
        (*self).get(index)
    }
}

impl<Q, T> Get<Q> for [T]
where
    Q: core::slice::SliceIndex<[T]>,
{
    type Key = usize;
    type Output = Q::Output;

    fn get(&self, index: Q) -> Option<&Self::Output>
    where
        Self::Key: core::borrow::Borrow<Q>,
    {
        self.as_ref().get(index)
    }
}

#[cfg(feature = "hashbrown")]
impl<Q, K, V, S> Get<Q> for hashbrown::HashMap<K, V, S>
where
    Q: Eq + core::hash::Hash,
    K: Eq + core::hash::Hash,
    S: core::hash::BuildHasher,
{
    type Key = K;
    type Output = V;

    fn get(&self, index: Q) -> Option<&V>
    where
        Self::Key: core::borrow::Borrow<Q>,
    {
        self.get(&index)
    }
}
