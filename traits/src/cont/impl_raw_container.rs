/*
    Appellation: impl_store <module>
    Created At: 2025.12.10:21:32:17
    Contrib: @FL03
*/
use super::RawContainer;

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

impl<T> RawContainer for core::slice::Iter<'_, T> {
    type Elem = T;
}

macro_rules! impl_raw_container  {
    (impl<Elem = $elem:ident> $trait:ident for {$($($cont:ident)::*<$($T:ident),*> $({where $($rest:tt)*})?),* $(,)?} ) => {
        $(impl_raw_container! {
            @impl<Elem = $elem> $trait for $($cont)::*<$($T),*> $(where $($rest)*)?
        })*
    };
    (@impl<Elem = $elem:ident> $trait:ident for $($cont:ident)::*<$($T:ident),*> $(where $($rest:tt)*)?) => {
        impl<$($T),*> $trait for $($cont)::*<$($T),*> $(where $($rest)*)? {
            type Elem = $elem;
        }
    };
}

impl_raw_container! {
    impl<Elem = T> RawContainer for {
        core::option::Option<T>,
        core::cell::UnsafeCell<T>,
        core::ops::Range<T>,
        core::result::Result<T, E>,
    }
}

#[cfg(feature = "alloc")]
impl_raw_container! {
    impl<Elem = T> RawContainer for {
        alloc::boxed::Box<T>,
        alloc::rc::Rc<T>,
        alloc::sync::Arc<T>,
        alloc::vec::Vec<T>,
        alloc::collections::BTreeSet<T>,
        alloc::collections::LinkedList<T>,
        alloc::collections::VecDeque<T>,
        alloc::collections::BinaryHeap<T>,
        alloc::collections::BTreeMap<K, T>,
    }
}

#[cfg(feature = "std")]
impl_raw_container! {
    impl<Elem = T> RawContainer for {
        std::cell::Cell<T>,
        std::cell::OnceCell<T>,
        std::cell::RefCell<T>,
        std::sync::Mutex<T>,
        std::sync::RwLock<T>,
        std::sync::LazyLock<T>,
        std::collections::HashMap<K, T>,
        std::collections::HashSet<T>,
    }
}

impl<A, S, D> RawContainer for ndarray::ArrayBase<S, D, A>
where
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = A>,
{
    type Elem = A;
}

#[cfg(feature = "hashbrown")]
impl<K, V, S> RawContainer for hashbrown::HashMap<K, V, S> {
    type Elem = V;
}
#[cfg(feature = "hashbrown")]
impl<K, S> RawContainer for hashbrown::HashSet<K, S> {
    type Elem = K;
}
