/*
    Appellation: stores <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[cfg(no_std)]
use alloc::collections::{btree_map, BTreeMap};
#[cfg(not(no_std))]
use std::collections::{btree_map, hash_map, BTreeMap, HashMap};

pub trait Entry<'a> {
    type Key;
    type Value;

    fn key(&self) -> &Self::Key;

    fn or_insert(self, default: Self::Value) -> &'a mut Self::Value;
}

pub trait OrInsert<K, V> {
    fn or_insert(&mut self, key: K, value: V) -> &mut V;
}

pub trait Store<K, V> {
    fn get(&self, key: &K) -> Option<&V>;

    fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    fn insert(&mut self, key: K, value: V) -> Option<V>;

    fn remove(&mut self, key: &K) -> Option<V>;
}

macro_rules! entry {
    ($($prefix:ident)::* -> $call:ident($($arg:tt),*)) => {
        $($prefix)::*::Entry::$call($($arg),*)
    };

}

macro_rules! impl_entry {
    ($($prefix:ident)::* where $($preds:tt)* ) => {

        impl<'a, K, V> Entry<'a> for $($prefix)::*::Entry<'a, K, V> where $($preds)* {
            type Key = K;
            type Value = V;

            fn key(&self) -> &Self::Key {
                entry!($($prefix)::* -> key(self))
            }

            fn or_insert(self, default: Self::Value) -> &'a mut Self::Value {
                entry!($($prefix)::* -> or_insert(self, default))
            }
        }

    };

}

macro_rules! impl_store {
    ($t:ty, where $($preds:tt)* ) => {

        impl<K, V> Store<K, V> for $t where $($preds)* {
            fn get(&self, key: &K) -> Option<&V> {
                <$t>::get(self, &key)
            }

            fn get_mut(&mut self, key: &K) -> Option<&mut V> {
                <$t>::get_mut(self, &key)
            }

            fn insert(&mut self, key: K, value: V) -> Option<V> {
                <$t>::insert(self, key, value)
            }

            fn remove(&mut self, key: &K) -> Option<V> {
                <$t>::remove(self, &key)
            }
        }

    };
}

impl_entry!(btree_map where K: Ord);
#[cfg(feature = "std")]
impl_entry!(hash_map where K: Eq + core::hash::Hash);
impl_store!(BTreeMap<K, V>, where K: Ord);
#[cfg(feature = "std")]
impl_store!(HashMap<K, V>, where K: Eq + core::hash::Hash);
