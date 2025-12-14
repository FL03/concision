/*
    Appellation: containers <module>
    Created At: 2025.12.10:21:29:49
    Contrib: @FL03
*/
use super::DataContainer;

macro_rules! impl_data_container {
    ($($($container:ident)::*<$A:ident $(, $B:ident)?>),* $(,)?) => {
        $(impl_data_container! {
            @impl<$A $(, $B)?> $($container)::*
        })*
    };
    (@impl<$T:ident> $($container:ident)::*) => {
        impl<$T> $crate::cont::DataContainer<$T> for $($container)::*<$T> {
            type Cont<U> = $($container)::*<U>;
        }
    };
    (@impl<$K:ident, $V:ident> $($container:ident)::*) => {
        impl<$K, $V>  $crate::cont::DataContainer<$V> for $($container)::*<$K, $V> {
            type Cont<U> = $($container)::*<$K, U>;
        }
    };
}

impl<T> DataContainer<T> for [T] {
    type Cont<U> = [U];
}

impl<T, E> DataContainer<T> for core::result::Result<T, E> {
    type Cont<U> = core::result::Result<U, E>;
}

impl_data_container! {
    core::option::Option<T>,
}

#[cfg(feature = "alloc")]
impl_data_container! {
    alloc::boxed::Box<T>,
    alloc::vec::Vec<T>,
    alloc::collections::BTreeMap<K, V>,
    alloc::collections::BTreeSet<K>,
    alloc::collections::VecDeque<T>,
    alloc::rc::Rc<T>,
    alloc::sync::Arc<T>,
}

#[cfg(feature = "std")]
impl_data_container! {
    std::collections::HashMap<K, V>,
    std::collections::HashSet<K>,
    std::cell::Cell<T>,
    std::sync::Mutex<T>,
    std::sync::RwLock<T>,
}
