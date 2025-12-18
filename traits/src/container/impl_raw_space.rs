/*
    Appellation: impl_store <module>
    Created At: 2025.12.10:21:32:17
    Contrib: @FL03
*/
use crate::container::RawSpace;

impl<S, T> RawSpace for &S
where
    S: RawSpace<Elem = T>,
{
    type Elem = T;
}

impl<S, T> RawSpace for &mut S
where
    S: RawSpace<Elem = T>,
{
    type Elem = T;
}

impl<T> RawSpace for [T] {
    type Elem = T;
}

impl<T, const N: usize> RawSpace for [T; N] {
    type Elem = T;
}

impl<T> RawSpace for core::slice::Iter<'_, T> {
    type Elem = T;
}

macro_rules! impl_raw_space  {
    (impl<Elem = $elem:ident> $trait:ident for {$(
        $($cont:ident)::*<$($T:ident),*> $({where $($rest:tt)*})?
    ),* $(,)?}) => {
        $(impl_raw_space! {
            @impl<Elem = $elem> $trait for $($cont)::*<$($T),*> $(where $($rest)*)?
        })*
    };
    (@impl<Elem = $elem:ident> $trait:ident for $($cont:ident)::*<$($T:ident),*> $(where $($rest:tt)*)?) => {
        impl<$($T),*> $trait for $($cont)::*<$($T),*> $(where $($rest)*)? {
            type Elem = $elem;
        }
    };
}

impl_raw_space! {
    impl<Elem = T> RawSpace for {
        core::option::Option<T>,
        core::cell::UnsafeCell<T>,
        core::ops::Range<T>,
        core::result::Result<T, E>,
        ndarray::ArrayBase<S, D, T> {
            where
                S: ndarray::RawData<Elem = T>,
                D: ndarray::Dimension
        },
    }
}

#[cfg(feature = "alloc")]
impl_raw_space! {
    impl<Elem = T> RawSpace for {
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
impl_raw_space! {
    impl<Elem = T> RawSpace for {
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

#[cfg(feature = "hashbrown")]
impl_raw_space! {
    impl<Elem = T> RawSpace for {
        hashbrown::HashMap<K, T, S>,
        hashbrown::HashSet<T, S>,
    }
}
