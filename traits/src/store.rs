/*
    Appellation: key_value <module>
    Created At: 2026.01.13:20:49:22
    Contrib: @FL03
*/
/// The [`StoreEntry`] trait establishes a common interface for all _entries_ within a
/// key-value store. These types enable in-place manipulation of key-value pairs by allowing
/// for keys to point to empty or _vacant_ slots within the store.
pub trait StoreEntry<'a, K, V> {
    /// checks if the entry is occupied
    fn is_occupied(&self) -> bool;
    /// checks if the entry is vacant
    fn is_vacant(&self) -> bool;
}

/// The [`RawStore`] trait is used to define an interface for key-value stores like hash-maps,
/// dictionaries, and similar data structures.
pub trait RawStore<K, V> {
    /// retrieves a reference to a value by key
    fn get(&self, key: &K) -> Option<&V>;
    /// returns true if the key is associated with a value in the store
    fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}
/// [`RawStoreMut`] extends the [`RawStore`] trait by introducing various mutable operations
/// and accessors for elements within the store.
pub trait RawStoreMut<K, V>: RawStore<K, V> {
    /// retrieves a mutable reference to a value by key
    fn get_mut(&mut self, key: &K) -> Option<&mut V>;
    /// inserts a key-value pair into the store
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    /// removes a key-value pair from the store by key
    fn remove(&mut self, key: &K) -> Option<V>;
}
/// The [`Store`] trait is a more robust interface for key-value stores, building upon both
/// [`RawStore`] and [`RawStoreMut`] traits by introducing an `entry` method for in-place
/// manipulation of key-value pairs.
pub trait Store<K, V>: RawStoreMut<K, V> {
    type Entry<'a>: StoreEntry<'a, K, V>
    where
        Self: 'a;
    /// returns the entry corresponding to the given key for in-place manipulation
    fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a>;
}
/*
 ************* Implementations *************
*/

// macro_rules! impl_store {
//     (<$($src:ident)::*>::$store:ident<$K:ident, $V:ident $(, $($T:ident),*)?> $(where $($where:tt)*)?) => {

//         impl<$K, $V $(, $($T),*)?> RawStore<$K, $V> for $($src)::*::$store<$K, $V $(, $($T),*)?>  {
//             fn contains_key(&self, key: &K) -> bool {
//                 $($src)::*::$store::contains_key(self, key)
//             }

//             fn get(&self, key: &K) -> Option<&V> {
//                 $($src)::*::$store::get(self, key)
//             }
//         }

//         impl<$K, $V $(, $($T),*)?> RawStoreMut<K, V> for $($src)::*::$store<$K, $V $(, $($T),*)?> $(where $($where)*)? {
//             fn insert(&mut self, key: K, value: V) -> Option<V> {
//                 $($src)::*::$store::insert(self, key, value)
//             }

//             fn get_mut(&mut self, key: &K) -> Option<&mut V> {
//                 $($src)::*::$store::get_mut(self, key)
//             }

//             fn remove(&mut self, key: &K) -> Option<V> {
//                 $($src)::*::$store::remove(self, key)
//             }
//         }

//         impl<$K, $V $(, $($T),*)?> Store<K, V> for $($src)::*::$store<$K, $V $(, $($T),*)?> $(where $($where)*)? {
//             type Entry<'a>
//                 = $($src)::*::Entry<'a, K, V>
//             where
//                 Self: 'a;

//             fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
//                 $($src)::*::$store::entry(self, key)
//             }
//         }
//     };
// }

// #[cfg(feature = "alloc")]
// impl_store! {
//     <alloc::collections::btree_map>::BTreeMap<K, V> where K: Ord
// }

// #[cfg(feature = "hashbrown")]
// impl_store! {
//     <hashbrown::hash_map>::HashMap<K, V, S> where K: Eq + core::hash::Hash, S: core::hash::BuildHasher,
// }

// #[cfg(feature = "std")]
// impl_store! {
//     <std::collections::hash_map>::HashMap<K, V> where K: Eq + core::hash::Hash,
// }

#[cfg(feature = "alloc")]
mod impl_alloc {
    use super::*;

    use alloc::collections::btree_map::{self, BTreeMap};

    impl<'a, K, V> StoreEntry<'a, K, V> for btree_map::Entry<'a, K, V> {
        fn is_occupied(&self) -> bool {
            matches!(self, btree_map::Entry::Occupied(_))
        }

        fn is_vacant(&self) -> bool {
            matches!(self, btree_map::Entry::Vacant(_))
        }
    }

    impl<K, V> RawStore<K, V> for BTreeMap<K, V>
    where
        K: Ord,
    {
        fn contains_key(&self, key: &K) -> bool {
            BTreeMap::contains_key(self, key)
        }

        fn get(&self, key: &K) -> Option<&V> {
            BTreeMap::get(self, key)
        }
    }

    impl<K, V> Store<K, V> for BTreeMap<K, V>
    where
        K: Ord,
    {
        type Entry<'a>
            = btree_map::Entry<'a, K, V>
        where
            Self: 'a;

        fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
            BTreeMap::entry(self, key)
        }
    }

    impl<K, V> RawStoreMut<K, V> for BTreeMap<K, V>
    where
        K: Ord,
    {
        fn insert(&mut self, key: K, value: V) -> Option<V> {
            BTreeMap::insert(self, key, value)
        }

        fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            BTreeMap::get_mut(self, key)
        }

        fn remove(&mut self, key: &K) -> Option<V> {
            BTreeMap::remove(self, key)
        }
    }
}

#[cfg(feature = "hashbrown")]
mod impl_hashbrown {
    use super::*;
    use core::hash::{BuildHasher, Hash};
    use hashbrown::hash_map::{self, HashMap};

    impl<K, V, S> StoreEntry<'_, K, V> for hash_map::Entry<'_, K, V, S> {
        fn is_occupied(&self) -> bool {
            matches!(self, hash_map::Entry::Occupied(_))
        }

        fn is_vacant(&self) -> bool {
            matches!(self, hash_map::Entry::Vacant(_))
        }
    }

    impl<K, V, S> RawStore<K, V> for HashMap<K, V, S>
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        fn contains_key(&self, key: &K) -> bool {
            HashMap::contains_key(self, key)
        }

        fn get(&self, key: &K) -> Option<&V> {
            HashMap::get(self, key)
        }
    }

    impl<K, V, S> RawStoreMut<K, V> for HashMap<K, V, S>
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        fn insert(&mut self, key: K, value: V) -> Option<V> {
            HashMap::insert(self, key, value)
        }

        fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            HashMap::get_mut(self, key)
        }

        fn remove(&mut self, key: &K) -> Option<V> {
            HashMap::remove(self, key)
        }
    }

    impl<K, V, S> Store<K, V> for HashMap<K, V, S>
    where
        K: Eq + Hash,
        S: BuildHasher,
    {
        type Entry<'a>
            = hash_map::Entry<'a, K, V, S>
        where
            Self: 'a;

        fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
            HashMap::entry(self, key)
        }
    }
}

#[cfg(feature = "std")]
mod impl_std {
    use super::*;
    use core::hash::Hash;
    use std::collections::hash_map::{self, HashMap};

    impl<K, V> StoreEntry<'_, K, V> for hash_map::Entry<'_, K, V> {
        fn is_occupied(&self) -> bool {
            matches!(self, hash_map::Entry::Occupied(_))
        }

        fn is_vacant(&self) -> bool {
            matches!(self, hash_map::Entry::Vacant(_))
        }
    }

    impl<K, V> RawStore<K, V> for HashMap<K, V>
    where
        K: Eq + Hash,
    {
        fn contains_key(&self, key: &K) -> bool {
            HashMap::contains_key(self, key)
        }

        fn get(&self, key: &K) -> Option<&V> {
            HashMap::get(self, key)
        }
    }

    impl<K, V> RawStoreMut<K, V> for HashMap<K, V>
    where
        K: Eq + Hash,
    {
        fn insert(&mut self, key: K, value: V) -> Option<V> {
            HashMap::insert(self, key, value)
        }
        fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            HashMap::get_mut(self, key)
        }

        fn remove(&mut self, key: &K) -> Option<V> {
            HashMap::remove(self, key)
        }
    }

    impl<K, V> Store<K, V> for HashMap<K, V>
    where
        K: Eq + Hash,
    {
        type Entry<'a>
            = hash_map::Entry<'a, K, V>
        where
            Self: 'a;

        fn entry<'a>(&'a mut self, key: K) -> Self::Entry<'a> {
            HashMap::entry(self, key)
        }
    }
}
