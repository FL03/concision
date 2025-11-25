/*
    Appellation: event <module>
    Created At: 2025.11.25:09:25:50
    Contrib: @FL03
*/

/// A simple synaptic event: weight added to synaptic variable `s` when it arrives.

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SynapticEvent<T = f64> {
    /// instantaneous weight added to synaptic variable `s`.
    pub weight: T,
}

impl<T> SynapticEvent<T> {
    /// Create a new SynapticEvent
    pub const fn new(weight: T) -> Self {
        Self { weight }
    }
}
