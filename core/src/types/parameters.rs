/*
    Appellation: parameters <module>
    Created At: 2025.12.14:07:14:46
    Contrib: @FL03
*/

pub trait HyperParamKey:
    Eq + AsRef<str> + core::borrow::Borrow<str> + core::fmt::Debug + core::hash::Hash
{
    private! {}
}

impl HyperParamKey for str {
    seal! {}
}

#[cfg(feature = "alloc")]
impl HyperParamKey for alloc::string::String {
    seal! {}
}

/// The [`Parameter`] struct represents a key-value pair used for configuration
/// settings within the neural network framework.
pub struct Parameter<K, V> {
    pub key: K,
    pub value: V,
}

impl<K, V> Parameter<K, V> {
    /// creates a new parameter instance
    pub const fn new(key: K, value: V) -> Self {
        Self { key, value }
    }
    /// returns a reference to the key
    pub const fn key(&self) -> &K {
        &self.key
    }
    /// returns a reference to the value
    pub const fn value(&self) -> &V {
        &self.value
    }
    /// returns a mutable reference to the value
    pub fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}
