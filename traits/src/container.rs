/*
    Appellation: container <module>
    Created At: 2025.11.28:15:30:46
    Contrib: @FL03
*/
/// The [`Container`] trait defines a generic interface for container types.
pub trait Container {
    type Cont<U>: ?Sized;
    type Item;
}

pub trait KeyValue {
    type Cont<_K, _V>;
    type Key;
    type Value;

    fn key(&self) -> &Self::Key;
    fn value(&self) -> &Self::Value;
}

impl<K, V> KeyValue for (K, V) {
    type Cont<_K, _V> = (_K, _V);
    type Key = K;
    type Value = V;

    fn key(&self) -> &Self::Key {
        &self.0
    }

    fn value(&self) -> &Self::Value {
        &self.1
    }
}

macro_rules! container {
    ($(
        $($container:ident)::*<$A:ident $(, $B:ident)?>
    ),* $(,)?) => {
        $(container!(@impl $($container)::*<$A $(, $B)?>);)*
    };

    (@impl $($container:ident)::*<$T:ident>) => {
        impl<$T> $crate::container::Container for $($container)::*<$T> {
            type Cont<U> = $($container)::*<U>;
            type Item = $T;
        }
    };
    (@impl $($container:ident)::*<$K:ident, $V:ident>) => {
        impl<$K, $V>  $crate::container::Container for $($container)::*<$K, $V> {
            type Cont<U> = $($container)::*<$K, U>;
            type Item = $V;
        }
    };
}

impl<T> Container for [T] {
    type Cont<U> = [U];
    type Item = T;
}

container! {
    core::option::Option<T>,
    core::result::Result<T, E>,
}

#[cfg(feature = "alloc")]
container! {
    alloc::boxed::Box<T>,
    alloc::vec::Vec<T>,
    alloc::collections::BTreeMap<K, V>,
    alloc::collections::BTreeSet<K>,
    alloc::collections::VecDeque<T>,
    alloc::rc::Rc<T>,
    alloc::sync::Arc<T>,
}

#[cfg(feature = "std")]
container! {
    std::collections::HashMap<K, V>,
    std::collections::HashSet<K>,
    std::cell::Cell<T>,
    std::sync::Mutex<T>,
    std::sync::RwLock<T>,
}
