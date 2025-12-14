/*
    Appellation: impl_store <module>
    Created At: 2025.12.10:21:32:17
    Contrib: @FL03
*/
use super::RawContainer;

macro_rules! impl_raw_container {
    (@impl $($name:ident)::*<$T:ident>) => {
        impl<$T> RawContainer for $($name)::*<$T> {
            type Elem = $T;
        }
    };
    {$($($name:ident)::*<$T:ident>),* $(,)?} => {
        $(impl_raw_container!(@impl $($name)::*<$T>);)*
    };
}

impl<S, T> RawContainer for &S
where
    S: RawContainer<Elem = T>,
{
    type Elem = T;
}

impl<S, T> RawContainer for &mut S
where
    S: RawContainer<Elem = T>,
{
    type Elem = T;
}

impl<T> RawContainer for [T] {
    type Elem = T;
}

impl<T, const N: usize> RawContainer for [T; N] {
    type Elem = T;
}

impl<T, E> RawContainer for core::result::Result<T, E> {
    type Elem = T;
}

impl_raw_container! {
    Option<T>,
}

#[cfg(all(feature = "alloc", not(feature = "nightly")))]
impl_raw_container! {
    alloc::vec::Vec<T>,
    alloc::boxed::Box<T>,
    alloc::rc::Rc<T>,
    alloc::rc::Weak<T>,
    alloc::sync::Arc<T>,
    alloc::collections::BinaryHeap<T>,
    alloc::collections::BTreeSet<T>,
    alloc::collections::LinkedList<T>,
    alloc::collections::VecDeque<T>,
}

#[cfg(feature = "std")]
impl_raw_container! {
    std::cell::Cell<T>,
    std::cell::OnceCell<T>,
    std::cell::RefCell<T>,
    std::sync::Mutex<T>,
    std::sync::RwLock<T>,
    std::sync::LazyLock<T>,
}

#[cfg(all(feature = "alloc", not(feature = "nightly")))]
impl<K, V> RawContainer for alloc::collections::BTreeMap<K, V> {
    type Elem = V;
}

#[cfg(feature = "std")]
impl<K, V, S> RawContainer for std::collections::HashMap<K, V, S> {
    type Elem = V;
}
#[cfg(feature = "std")]
impl<K, S> RawContainer for std::collections::HashSet<K, S> {
    type Elem = K;
}

#[cfg(feature = "hashbrown")]
impl<K, V, S> RawContainer for hashbrown::HashMap<K, V, S> {
    type Elem = V;
}
#[cfg(feature = "hashbrown")]
impl<K, S> RawContainer for hashbrown::HashSet<K, S> {
    type Elem = K;
}

#[cfg(all(feature = "alloc", feature = "nightly"))]
mod impl_alloc {
    use crate::store::RawContainer;
    use alloc::alloc::Allocator;
    use alloc::boxed::Box;
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::vec::Vec;

    impl<T, A> RawContainer for Box<T, A>
    where
        A: Allocator + Clone,
    {
        type Elem = T;
    }

    impl<K, V, A> RawContainer for BTreeMap<K, V, A>
    where
        A: Allocator + Clone,
    {
        type Elem = V;
    }

    impl<K, A> RawContainer for BTreeSet<K, A>
    where
        A: Allocator + Clone,
    {
        type Elem = K;
    }

    impl<T, A> RawContainer for Vec<T, A>
    where
        A: Allocator + Clone,
    {
        type Elem = T;
    }

    #[cfg(feature = "hashbrown")]
    impl<K, V, S, A> RawContainer for hashbrown::HashMap<K, V, S, A>
    where
        A: Allocator,
    {
        type Elem = V;
    }
    #[cfg(feature = "hashbrown")]
    impl<K, S, A> RawContainer for hashbrown::HashSet<K, S, A>
    where
        A: Allocator,
    {
        type Elem = K;
    }
}
