/*
    Appellation: utils <module>
    Created At: 2025.12.08:15:53:49
    Contrib: @FL03
*/
use super::{Leaky, SynapticEvent};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use num_traits::{Float, FromPrimitive, NumAssign};

/// Compute the minimum external drive required to make a leaky integrate-and-fire neuron spike
/// over a single time step. This is based on the analytical solution of the leaky
/// integrate-and-fire model.
#[inline]
pub fn get_min_drive<T: Float>(v_rest: T, v_th: T, tau_m: T, dt: T) -> T {
    let exp_term = (-dt / tau_m).exp();
    let numerator = v_th - v_rest * exp_term;
    let denominator = T::one() - exp_term;
    numerator / denominator
}

/// A basic method for _discovering_ the minimum external drive required to make a spiking
/// neuron spike
#[inline]
pub fn sweep_for_min_drive<T>(step_size: T) -> T
where
    T: Float + FromPrimitive + NumAssign,
{
    let dt = T::from_f32(0.1).unwrap();
    let t_sim = T::from_usize(1000).unwrap();
    let steps = (t_sim / dt).to_usize().unwrap();
    let presyn_spikes: Vec<(T, T)> = vec![]; // no extra synaptic drive

    let mut i_ext = T::zero();
    loop {
        let mut neuron = Leaky::<T>::default();
        let mut events: Vec<Vec<SynapticEvent<T>>> = vec![Vec::new(); steps + 1];
        for (t_spike, weight) in &presyn_spikes {
            let idx = (*t_spike / dt).round().to_isize().unwrap();
            if idx >= 0 && (idx as usize) < events.len() {
                events[idx as usize].push(SynapticEvent { weight: *weight });
            }
        }
        let mut spiked = false;
        for step in 0..steps {
            for ev in &events[step] {
                neuron.apply_spike(ev.weight);
            }
            let res = neuron.step(dt, i_ext);
            if res.is_spiked() {
                spiked = true;
                break;
            }
        }
        if spiked {
            break;
        }
        i_ext += step_size; // increment drive
    }
    i_ext
}
