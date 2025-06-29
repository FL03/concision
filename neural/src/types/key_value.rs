/*
    appellation: key_value <module>
    authors: @FL03
*/

/// The [`KeyValue`] type is used to generically represent a simple key-value pair within a
/// store.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct KeyValue<K = String, V = f64> {
    pub key: K,
    pub value: V,
}

impl<K, V> KeyValue<K, V> {
    pub const fn new(key: K, value: V) -> Self {
        Self { key, value }
    }
    /// returns a new [`KeyValue`] from the given key, using the logical default for the value
    pub fn from_key(key: K) -> Self
    where
        V: Default,
    {
        Self {
            key,
            value: V::default(),
        }
    }
    /// returns a new [`KeyValue`] from the given value, using the logical default for the key
    pub fn from_value(value: V) -> Self
    where
        K: Default,
    {
        Self {
            key: K::default(),
            value,
        }
    }
    /// returns an immutable reference to the key
    pub const fn key(&self) -> &K {
        &self.key
    }
    /// returns a mutable reference to the key
    pub const fn key_mut(&mut self) -> &mut K {
        &mut self.key
    }
    /// returns an immutable reference to the value
    pub const fn value(&self) -> &V {
        &self.value
    }
    /// returns a mutable reference to the value
    pub const fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
    /// update the current key and return a mutable reference to self
    pub fn set_key(&mut self, key: K) -> &mut Self {
        self.key = key;
        self
    }
    /// update the current value and return a mutable reference to self
    pub fn set_value(&mut self, value: V) -> &mut Self {
        self.value = value;
        self
    }
    /// consumes the current instance to create another with the given key
    pub fn with_key<K2>(self, key: K2) -> KeyValue<K2, V> {
        KeyValue {
            key,
            value: self.value,
        }
    }
    /// consumes the current instance to create another with the given value
    pub fn with_value<V2>(self, value: V2) -> KeyValue<K, V2> {
        KeyValue {
            key: self.key,
            value,
        }
    }
    /// [`replace`](core::mem::replace) the current value and return the old value
    pub const fn replace_value(&mut self, value: V) -> V {
        core::mem::replace(self.value_mut(), value)
    }
    /// [`swap`](core::mem::swap) the current value with another in the given instance
    pub const fn swap_value(&mut self, other: &mut KeyValue<K, V>) {
        core::mem::swap(self.value_mut(), other.value_mut())
    }
    /// [`take`](core::mem::take) the current value and return it, replacing it with the
    /// logical default
    pub fn take_value(&mut self) -> V
    where
        V: Default,
    {
        core::mem::take(self.value_mut())
    }
    /// returns a new instance of the [`KeyValue`] with mutable references to the value and a
    /// reference to the key
    pub fn entry(&mut self) -> KeyValue<&K, &mut V> {
        KeyValue {
            key: &self.key,
            value: &mut self.value,
        }
    }
    /// returns a new instance of the [`KeyValue`] with references to the key and value
    pub const fn view(&self) -> KeyValue<&K, &V> {
        KeyValue {
            key: self.key(),
            value: self.value(),
        }
    }
    /// returns a new instance of the [`KeyValue`] with mutable references to the current key
    /// and value
    pub const fn view_mut(&mut self) -> KeyValue<&mut K, &mut V> {
        KeyValue {
            key: &mut self.key,
            value: &mut self.value,
        }
    }
}

impl<K, V> KeyValue<&K, &V> {
    /// returns a new [`KeyValue`] instance with clones of the current key and value
    pub fn cloned(&self) -> KeyValue<K, V>
    where
        K: Clone,
        V: Clone,
    {
        KeyValue {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
    /// returns a new [`KeyValue`] instance with copies of the current key and value
    pub fn copied(&self) -> KeyValue<K, V>
    where
        K: Copy,
        V: Copy,
    {
        KeyValue {
            key: *self.key,
            value: *self.value,
        }
    }
}

impl<K, V> KeyValue<&K, &mut V> {
    /// returns a new [`KeyValue`] instance with clones of the current key and value
    pub fn cloned(&self) -> KeyValue<K, V>
    where
        K: Clone,
        V: Clone,
    {
        KeyValue {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
    /// returns a new [`KeyValue`] instance with copies of the current key and value
    pub fn copied(&self) -> KeyValue<K, V>
    where
        K: Copy,
        V: Copy,
    {
        KeyValue {
            key: *self.key,
            value: *self.value,
        }
    }
}

impl<K, V> KeyValue<&mut K, &mut V> {
    /// returns a new [`KeyValue`] instance with clones of the current key and value
    pub fn cloned(&self) -> KeyValue<K, V>
    where
        K: Clone,
        V: Clone,
    {
        KeyValue {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
    /// returns a new [`KeyValue`] instance with copies of the current key and value
    pub fn copied(&self) -> KeyValue<K, V>
    where
        K: Copy,
        V: Copy,
    {
        KeyValue {
            key: *self.key,
            value: *self.value,
        }
    }
}

impl<K, V> core::fmt::Display for KeyValue<K, V>
where
    K: core::fmt::Display,
    V: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{k}: {v}", k = self.key(), v = self.value())
    }
}
