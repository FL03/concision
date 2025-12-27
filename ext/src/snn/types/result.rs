/*
    Appellation: result <module>
    Created At: 2025.11.25:09:21:16
    Contrib: @FL03
*/

/// Result of a single integration step.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[repr(C)]
pub enum StepResult<T = f64> {
    Spiked { v: T },
    NotSpiked { v: T },
}

impl<T> StepResult<T> {
    /// create a new instance of the `StepResult` using a boolean to indicate if the neuron
    /// spiked during the step
    pub const fn new(v: T, spiked: bool) -> Self {
        if spiked {
            Self::Spiked { v }
        } else {
            Self::NotSpiked { v }
        }
    }
    /// returns an [`NotSpiked`](StepResult::NotSpiked) variant of the result with the given membrane potential
    pub const fn not_spiked(v: T) -> Self {
        Self::NotSpiked { v }
    }
    /// returns a [`Spiked`](StepResult::Spiked) variant of the result with the given membrane potential
    pub const fn spiked(v: T) -> Self {
        Self::Spiked { v }
    }
    #[inline]
    /// returns a new instance of the result configured as a _spiked_ variant.
    pub fn spike(self) -> Self {
        match self {
            Self::NotSpiked { v } => Self::Spiked { v },
            _ => self,
        }
    }
    #[inline]
    /// consumes the current instance to create another that is said to have _not spiked_.
    pub fn unspike(self) -> Self {
        match self {
            Self::Spiked { v } => Self::NotSpiked { v },
            _ => self,
        }
    }
    /// returns true if the result is of a [`Spiked`](StepResult::Spiked) variant
    pub const fn is_spiked(&self) -> bool {
        matches!(self, Self::Spiked { .. })
    }
    /// returns true if the result is of a [`NotSpiked`](StepResult::NotSpiked) variant
    pub const fn is_not_spiked(&self) -> bool {
        matches!(self, Self::NotSpiked { .. })
    }
    /// returns a reference to the membrane potential (`v`)
    pub const fn get(&self) -> &T {
        match self {
            Self::Spiked { v } => v,
            Self::NotSpiked { v } => v,
        }
    }
    /// returns a mutable reference to the membrane potential (`v`)
    pub const fn get_mut(&mut self) -> &mut T {
        match self {
            Self::Spiked { v } => v,
            Self::NotSpiked { v } => v,
        }
    }
}

impl<T> AsRef<T> for StepResult<T> {
    fn as_ref(&self) -> &T {
        self.get()
    }
}

impl<T> AsMut<T> for StepResult<T> {
    fn as_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> core::borrow::Borrow<T> for StepResult<T> {
    fn borrow(&self) -> &T {
        self.get()
    }
}

impl<T> core::borrow::BorrowMut<T> for StepResult<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> core::ops::Deref for StepResult<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> core::ops::DerefMut for StepResult<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<T> PartialEq<bool> for StepResult<T> {
    fn eq(&self, other: &bool) -> bool {
        &self.is_spiked() == other
    }
}
