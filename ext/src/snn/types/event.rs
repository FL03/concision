/*
    Appellation: event <module>
    Created At: 2025.11.25:09:25:50
    Contrib: @FL03
*/

/// A synaptic event that modifies the synaptic variable `s` by an instantaneous weight.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[repr(transparent)]
pub struct SynapticEvent<T = f64> {
    /// instantaneous weight added to synaptic variable `s`.
    pub weight: T,
}

impl<T> SynapticEvent<T> {
    /// returns a new instance of the `SynapticEvent` using the given weight
    pub const fn new(weight: T) -> Self {
        Self { weight }
    }
    /// returns a reference to the weight
    pub const fn weight(&self) -> &T {
        &self.weight
    }
    /// returns a mutable reference to the weight of the synaptic event
    pub const fn weight_mut(&mut self) -> &mut T {
        &mut self.weight
    }
    /// [`replace`](core::mem::replace) the weight with a new value, returning the old value.
    pub const fn replace_weight(&mut self, weight: T) -> T {
        core::mem::replace(self.weight_mut(), weight)
    }
    /// sets the weight to a new value
    pub fn set_weight(&mut self, weight: T) {
        self.weight = weight;
    }
    /// [`swap`](core::mem::swap) the weight with the weight of another synaptic event
    pub const fn swap_weight(&mut self, other: &mut SynapticEvent<T>) {
        core::mem::swap(self.weight_mut(), other.weight_mut());
    }
    /// [`take`](core::mem::take) the weight, leaving the default value in its place.
    pub fn take_weight(&mut self) -> T
    where
        T: Default,
    {
        core::mem::take(&mut self.weight)
    }
}

impl<T> AsRef<T> for SynapticEvent<T> {
    fn as_ref(&self) -> &T {
        self.weight()
    }
}

impl<T> AsMut<T> for SynapticEvent<T> {
    fn as_mut(&mut self) -> &mut T {
        self.weight_mut()
    }
}

impl<T> core::borrow::Borrow<T> for SynapticEvent<T> {
    fn borrow(&self) -> &T {
        self.weight()
    }
}

impl<T> core::borrow::BorrowMut<T> for SynapticEvent<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.weight_mut()
    }
}

impl<T> core::ops::Deref for SynapticEvent<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.weight()
    }
}

impl<T> core::ops::DerefMut for SynapticEvent<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.weight_mut()
    }
}
