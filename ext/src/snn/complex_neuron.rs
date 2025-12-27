/*
    Appellation: complex_neuron <module>
    Created At: 2025.12.14:14:17:56
    Contrib: @FL03
*/
//! A minimal complex-valued leaky integrate-and-fire oscillator with continuous-time dynamics
//! (rotation-stable, radial leakage):
//! ```math
//! \frac{dz}{dt}=(i * \omega - \gamma)\cdot z + I(t)
//! ```
//! - $`\omega`$ : intrinsic angular velocity (rad/s)
//! - $`\gamma`$ : radial decay rate (1/s), gamma > 0 ensures boundedness
//! - $`I(t)`$  : complex-valued input (excitation)
//!
//! Exact discrete update for constant input over a step dt:
//!
//! ```math
//! \begin{equation}\begin{aligned}
//! \phi &= e\cdot\Big((i\cdot\omega - \gamma)\cdot{dt}\Big)
//! z_{n+1} &= \phi\cdot{z_n} + I\cdot\Big(\frac{1 - \phi}{i\cdot\omega - \gamma}\Big)
//! \end{aligned}\end{equation}
//! ```
//! Phase-based spike condition (example policy): fire when the unwrapped phase crosses
//! $`\theta_{spike}^{+}`$ in the positive direction and magnitude $`|z| >= r_{thresh}`$. On spike,
//! radius may be reset to $`r_{reset}`$ while preserving phase.
//!
use ndarray::{ArrayBase, Dimension, RawData};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, FloatConst, FromPrimitive};

/// Parameters for the complex LIF oscillator.
#[derive(Copy, Clone, Debug)]
pub struct ComplexNeuronParams<T = f32> {
    pub omega: T,       // intrinsic angular velocity (rad/s)
    pub gamma: T,       // radial decay (1/s), must be > 0 for boundedness
    pub r_threshold: T, // minimum radius to allow a spike
    pub r_reset: T,     // radius after spike (if reset policy used)
    pub theta_spike: T, // phase angle (radians) at which to trigger a spike
}

/// Lightweight state for the neuron.
#[derive(Copy, Clone, Debug)]
pub struct ComplexNeuronState<T = f32> {
    pub z: Complex<T>,         // complex state z = r * e^{i theta}
    pub last_phase: Option<T>, // last observed phase (radians), used for crossing detection
}

/// Minimal complex-valued LIF neuron.
pub struct ComplexNeuron<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = Complex<A>>,
{
    pub params: ArrayBase<S, D, Complex<A>>,
    pub state: ComplexNeuronState<A>,
}

impl<T> Default for ComplexNeuronParams<T>
where
    Complex<T>: ComplexFloat<Real = T>,
    T: Float + FloatConst + FromPrimitive,
{
    fn default() -> Self {
        Self {
            omega: T::from_usize(2).unwrap() * <T>::PI(),
            gamma: T::one(),
            r_threshold: T::from_f32(1.0).unwrap(),
            r_reset: T::from_f32(0.05).unwrap(),
            theta_spike: <T>::PI(),
        }
    }
}
