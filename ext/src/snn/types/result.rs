/*
    Appellation: result <module>
    Created At: 2025.11.25:09:21:16
    Contrib: @FL03
*/

/// Result of a single integration step.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct StepResult<T = f64> {
    /// Whether the neuron emitted a spike on this step.
    pub(crate) spiked: bool,
    /// The membrane potential after the step (mV or arbitrary units).
    pub(crate) v: T,
}

impl<T> StepResult<T> {
    /// returns a new instance of the `StepResult`
    pub const fn new(spiked: bool, v: T) -> Self {
        Self { spiked, v }
    }

    pub const fn spiked(v: T) -> Self {
        Self { spiked: true, v }
    }

    pub const fn not_spiked(v: T) -> Self {
        Self { spiked: false, v }
    }

    pub const fn is_spiked(&self) -> bool {
        self.spiked
    }
    /// returns a reference to the membrane potential (`v`)
    pub const fn membrane_potential(&self) -> &T {
        &self.v
    }
}
