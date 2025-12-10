use super::LeakyState;

impl<T> LeakyState<T> {
    /// Create a new `LeakyState` with all state variables initialized to zero.
    pub const fn new(v: T, w: T, s: T) -> Self {
        Self { v, w, s }
    }
    /// create a new state initialize to 1
    pub fn one() -> Self
    where
        T: num_traits::One,
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
        T: num_traits::Zero,
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
    pub fn set_w(&mut self, w: T) {
        self.w = w
    }
    #[inline]
    /// set the membrane potential (`v`) to the given value
    pub fn set_v(&mut self, v: T) {
        self.v = v
    }
    #[inline]
    /// update the synaptic state (`s`) to the given value
    pub fn set_s(&mut self, s: T) {
        self.s = s
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
    pub fn reset(&mut self)
    where
        T: Default,
    {
        self.v = T::default();
        self.w = T::default();
        self.s = T::default();
    }
    /// update all state variables to the given values
    #[inline]
    pub fn update(&mut self, v: T, w: T, s: T) {
        self.set_w(w);
        self.set_v(v);
        self.set_s(s);
    }
    /// Apply a presynaptic spike event to the neuron; this increments the synaptic variable `s`
    /// by `weight` instantaneously (models delta spike arrival).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, level = "trace"))]
    pub fn apply_spike(&mut self, weight: T)
    where
        T: core::ops::AddAssign,
    {
        *self.s_mut() += weight;
    }
}
