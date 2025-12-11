/*
    Appellation: impl_leaky_state <module>
    Created At: 2025.12.10:13:08:07
    Contrib: @FL03
*/
use super::LeakyState;
use num_traits::{One, Zero};

impl<T> LeakyState<T> {
    /// Create a new `LeakyState` with all state variables initialized to zero.
    pub const fn new(v: T, w: T, s: T) -> Self {
        Self { v, w, s }
    }
    /// returns a new instance of the leak state with the given membrane potential
    pub fn from_v(v: T) -> Self
    where
        T: Zero,
    {
        Self {
            v,
            w: T::zero(),
            s: T::zero(),
        }
    }
    /// create a new state initialize to 1
    pub fn one() -> Self
    where
        T: One,
    {
        Self {
            v: T::one(),
            w: T::one(),
            s: T::one(),
        }
    }
    /// create a new state initialize to zero
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self {
            v: T::zero(),
            w: T::zero(),
            s: T::zero(),
        }
    }
    /// returns a reference to the neuron's adaptation variable (`w`)
    pub const fn w(&self) -> &T {
        &self.w
    }
    /// returns a mutable reference to the neuron's adaptation variable (`w`)
    pub const fn w_mut(&mut self) -> &mut T {
        &mut self.w
    }
    /// returns a reference to the membrane potential, `v`, of the neuron
    pub const fn v(&self) -> &T {
        &self.v
    }
    /// returns a mutable reference to the membrane potential, `v`, of the neuron
    pub const fn v_mut(&mut self) -> &mut T {
        &mut self.v
    }
    /// returns a reference to the current value, or synaptic state, of the neuron
    pub const fn s(&self) -> &T {
        &self.s
    }
    /// returns a mutable reference to the current value, or synaptic state, of the neuron
    pub const fn s_mut(&mut self) -> &mut T {
        &mut self.s
    }
    /// [`replace`](core::mem::replace) the values of one state with another returning the previous state
    pub fn replace(&mut self, other: LeakyState<T>) -> LeakyState<T> {
        LeakyState {
            v: core::mem::replace(&mut self.v, other.v),
            w: core::mem::replace(&mut self.w, other.w),
            s: core::mem::replace(&mut self.s, other.s),
        }
    }
    /// [`swap`](core::mem::swap) the values of one state with another
    pub const fn swap(&mut self, other: &mut LeakyState<T>) {
        core::mem::swap(&mut self.v, &mut other.v);
        core::mem::swap(&mut self.w, &mut other.w);
        core::mem::swap(&mut self.s, &mut other.s);
    }
    #[inline]
    /// [`take`](core::mem::take) the values of one state, replacing them with their logical defaults
    pub fn take(&mut self) -> Self
    where
        T: Default,
    {
        Self {
            v: core::mem::take(&mut self.v),
            w: core::mem::take(&mut self.w),
            s: core::mem::take(&mut self.s),
        }
    }
    #[inline]
    /// set the adaptation variable (`w`) to the given value
    pub fn set_w(&mut self, w: T) -> &mut Self {
        self.w = w;
        self
    }
    #[inline]
    /// set the membrane potential (`v`) to the given value
    pub fn set_v(&mut self, v: T) -> &mut Self {
        self.v = v;
        self
    }
    #[inline]
    /// update the synaptic state (`s`) to the given value
    pub fn set_s(&mut self, s: T) -> &mut Self {
        self.s = s;
        self
    }
    #[inline]
    /// consumes the current instance to create another with the given adaptation (`w`)
    pub fn with_w(self, w: T) -> Self {
        Self { w, ..self }
    }
    #[inline]
    /// consumes the current instance to create another with the given membrane potential (`v`)
    pub fn with_v(self, v: T) -> Self {
        Self { v, ..self }
    }
    #[inline]
    /// consumes the current instance to create another with the given synaptic state (`s`)
    pub fn with_s(self, s: T) -> Self {
        Self { s, ..self }
    }
    /// reset all state variables to their logical defaults
    #[inline]
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, target = "leaky", name = "leaky_state::reset")
    )]
    pub fn reset(&mut self)
    where
        T: Default,
    {
        #[cfg(feature = "tracing")]
        tracing::trace!("Resetting leaky neuron state to default values.");
        self.v = T::default();
        self.w = T::default();
        self.s = T::default();
    }
    /// update all state variables to the given values
    #[inline]
    pub fn update(&mut self, v: T, w: T, s: T) {
        self.set_w(w).set_v(v).set_s(s);
    }
    /// Apply an incrementation to the adaptation variable `w` of the neuron.
    pub fn apply_adaptation(&mut self, dw: T) -> &mut Self
    where
        T: core::ops::AddAssign,
    {
        self.w += dw;
        self
    }
    /// Apply the given increment to the synaptic variable `s` of the neuron.
    pub fn apply_spike(&mut self, ds: T) -> &mut Self
    where
        T: core::ops::AddAssign,
    {
        self.s += ds;
        self
    }
    /// returns a reference to the value associated with the given key
    ///
    /// # Panics
    ///
    /// Panics if the key is not one of "v", "w", "s", or one of their respective aliases.
    pub fn get(&self, key: &str) -> &T {
        match key {
            "v" | "membrane_potential" => &self.v,
            "w" | "adaptation_variable" => &self.w,
            "s" | "synaptic_current" => &self.s,
            _ => panic!("invalid key for LeakyState: {}", key),
        }
    }
    /// returns a reference to the value associated with the given key
    ///
    /// # Panics
    ///
    /// Panics if the key is not one of "v", "w", "s", or one of their respective aliases.
    pub fn get_mut(&mut self, key: &str) -> &mut T {
        match key {
            "v" | "membrane_potential" => &mut self.v,
            "w" | "adaptation_variable" => &mut self.w,
            "s" | "synaptic_current" => &mut self.s,
            _ => panic!("invalid key for LeakyState: {}", key),
        }
    }
}

impl<T> Default for LeakyState<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<T> core::ops::Index<usize> for LeakyState<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index % 3 {
            0 => &self.v,
            1 => &self.w,
            2 => &self.s,
            _ => unreachable!("modulo 3 should only yield 0, 1, or 2"),
        }
    }
}

impl<T> core::ops::Index<char> for LeakyState<T> {
    type Output = T;

    fn index(&self, index: char) -> &Self::Output {
        match index {
            'v' => &self.v,
            'w' => &self.w,
            's' => &self.s,
            _ => panic!("invalid index for LeakyState: {}", index),
        }
    }
}

impl<T> core::ops::Index<&str> for LeakyState<T> {
    type Output = T;

    fn index(&self, index: &str) -> &Self::Output {
        match index {
            "v" | "membrane_potential" => &self.v,
            "w" | "adaptation_variable" => &self.w,
            "s" | "synaptic_current" => &self.s,
            _ => panic!("invalid index for LeakyState: {}", index),
        }
    }
}
