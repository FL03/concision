//! Single spiking neuron (LIF + adaptation + exponential synapse) example in pure Rust.
//!
//! Model (forward-Euler integration; units are arbitrary but consistent):
//!   tau_m * dv/dt = -(v - v_rest) + R*(I_ext + I_syn) - w
//!   tau_w * dw/dt = -w
//!   tau_s * ds/dt = -s  (+ instantaneous increments when presynaptic spikes arrive)
//!
//! Spike: when v >= v_thresh -> spike emitted, v <- v_reset, w += b
//! I_syn = s
//!
//! The implementation is conservative with allocations and idiomatic Rust.

/// Result of a single integration step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    /// Whether the neuron emitted a spike on this step.
    pub spiked: bool,
    /// The membrane potential after the step (mV or arbitrary units).
    pub v: f64,
}

/// A simple synaptic event: weight added to synaptic variable `s` when it arrives.
#[derive(Debug, Clone, Copy)]
pub struct SynapticEvent {
    /// instantaneous weight added to synaptic variable `s`.
    pub weight: f64,
}

/// Leaky Integrate-and-Fire neuron with an adaptation term and exponential synaptic current.
///
/// All fields are public for convenience in research workflows; in production you may want to
/// expose read-only getters and safe setters only.
#[derive(Clone)]
pub struct SpikingNeuron {
    // ---- Parameters ----
    /// Membrane time constant `tau_m` (ms)
    pub tau_m: f64,
    /// Membrane resistance `R` (MΩ or arbitrary)
    pub resistance: f64,
    /// Resting potential `v_rest` (mV)
    pub v_rest: f64,
    /// Threshold potential `v_thresh` (mV)
    pub v_thresh: f64,
    /// Reset potential after spike `v_reset` (mV)
    pub v_reset: f64,

    /// Adaptation time constant `tau_w` (ms)
    pub tau_w: f64,
    /// Adaptation increment added on spike `b` (same units as w/current)
    pub b: f64,

    /// Synaptic time constant `tau_s` (ms)
    pub tau_s: f64,

    // ---- State variables ----
    /// Membrane potential `v`
    pub v: f64,
    /// Adaptation variable `w`
    pub w: f64,
    /// Synaptic variable `s` representing total synaptic current
    pub s: f64,

    // ---- Optional numerical safeguards ----
    /// Minimum allowed dt for integration (ms)
    pub min_dt: f64,
}

impl core::fmt::Debug for SpikingNeuron {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpikingNeuron")
            .field("tau_m", &self.tau_m)
            .field("resistance", &self.resistance)
            .field("v_rest", &self.v_rest)
            .field("v_thresh", &self.v_thresh)
            .field("v_reset", &self.v_reset)
            .field("tau_w", &self.tau_w)
            .field("b", &self.b)
            .field("tau_s", &self.tau_s)
            .field("v", &self.v)
            .field("w", &self.w)
            .field("s", &self.s)
            .finish()
    }
}

impl SpikingNeuron {
    /// Create a new neuron with common default parameters (units: ms and mV-like).
    ///
    /// Many fields are set to common neuroscience-like defaults but these are research parameters
    /// and should be tuned for your experiments.
    pub fn new_default() -> Self {
        let tau_m = 20.0; // ms
        let resistance = 1.0; // arbitrary
        let v_rest = -65.0; // mV
        let v_thresh = -50.0; // mV
        let v_reset = -65.0; // mV
        let tau_w = 200.0; // ms (slow adaptation)
        let b = 0.5; // adaptation increment
        let tau_s = 5.0; // ms (fast synapse)
        Self {
            tau_m,
            resistance,
            v_rest,
            v_thresh,
            v_reset,
            tau_w,
            b,
            tau_s,
            v: v_rest,
            w: 0.0,
            s: 0.0,
            min_dt: 1e-6,
        }
    }

    /// Create a neuron with explicit parameters and initial state.
    pub fn new(
        tau_m: f64,
        resistance: f64,
        v_rest: f64,
        v_thresh: f64,
        v_reset: f64,
        tau_w: f64,
        b: f64,
        tau_s: f64,
        initial_v: Option<f64>,
    ) -> Self {
        let v0 = initial_v.unwrap_or(v_rest);
        Self {
            tau_m,
            resistance,
            v_rest,
            v_thresh,
            v_reset,
            tau_w,
            b,
            tau_s,
            v: v0,
            w: 0.0,
            s: 0.0,
            min_dt: 1e-6,
        }
    }

    /// Reset state variables (keeps parameters).
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
        self.s = 0.0;
    }

    /// Apply a presynaptic spike event to the neuron; this increments the synaptic variable `s`
    /// by `weight` instantaneously (models delta spike arrival).
    pub fn receive_spike(&mut self, weight: f64) {
        self.s += weight;
    }

    /// Integrate the neuron state forward by `dt` milliseconds using forward Euler.
    ///
    /// `i_ext` is an externally injected current (same units as `s`).
    /// `dt` must be > 0.
    pub fn step(&mut self, dt: f64, i_ext: f64) -> StepResult {
        let dt = if dt <= 0.0 {
            panic!("dt must be > 0")
        } else {
            dt.max(self.min_dt)
        };

        // synaptic current is represented by `s`
        // ds/dt = -s / tau_s
        let ds = -self.s / self.tau_s;
        let s_next = self.s + dt * ds;

        // total synaptic current for this step (use current s, or average between s and s_next)
        // we use s for explicit Euler consistency.
        let i_syn = self.s;

        // membrane dv/dt = (-(v - v_rest) + R*(i_ext + i_syn) - w) / tau_m
        let dv =
            (-(self.v - self.v_rest) + self.resistance * (i_ext + i_syn) - self.w) / self.tau_m;
        let v_next = self.v + dt * dv;

        // adaptation dw/dt = -w / tau_w
        let dw = -self.w / self.tau_w;
        let w_next = self.w + dt * dw;

        // Commit state tentatively
        self.v = v_next;
        self.w = w_next;
        self.s = s_next;

        // Check for spike (simple threshold crossing)
        if self.v >= self.v_thresh {
            // spike: apply reset and adaptation increment
            self.v = self.v_reset;
            self.w += self.b;
            StepResult {
                spiked: true,
                v: self.v,
            }
        } else {
            StepResult {
                spiked: false,
                v: self.v,
            }
        }
    }

    /// Get current membrane potential
    pub fn membrane_potential(&self) -> f64 {
        self.v
    }

    /// Get current synaptic variable
    pub fn synaptic_state(&self) -> f64 {
        self.s
    }

    /// Get adaptation variable
    pub fn adaptation(&self) -> f64 {
        self.w
    }
}

impl Default for SpikingNeuron {
    fn default() -> Self {
                let tau_m = 20.0; // ms
        let resistance = 1.0; // arbitrary
        let v_rest = -65.0; // mV
        let v_thresh = -50.0; // mV
        let v_reset = -65.0; // mV
        let tau_w = 200.0; // ms (slow adaptation)
        let b = 0.5; // adaptation increment
        let tau_s = 5.0; // ms (fast synapse)
        Self {
            tau_m,
            resistance,
            v_rest,
            v_thresh,
            v_reset,
            tau_w,
            b,
            tau_s,
            v: v_rest,
            w: 0.0,
            s: 0.0,
            min_dt: 1e-6,
        }
    }
}

#[allow(dead_code)]
/// Minimal demonstration of neuron usage. Simulates a neuron for `t_sim` ms with dt,
/// injects a constant external current `i_ext`, and injects discrete synaptic events at specified times.
fn example() {
    // Simulation parameters
    let dt = 0.1; // ms
    let t_sim = 500.0; // ms
    let steps = (t_sim / dt) as usize;

    // Create neuron with defaults
    let mut neuron = SpikingNeuron::new_default();

    // Example external current (constant)
    let i_ext = 1.8; // tune to see spiking (units consistent with resistance & s)

    // Example presynaptic spike times (ms) and weights
    let presyn_spikes: Vec<(f64, f64)> =
        vec![(50.0, 2.0), (100.0, 1.5), (150.0, 2.2), (300.0, 3.0)];

    // Convert into an index-able event list
    let mut events: Vec<Vec<SynapticEvent>> = vec![Vec::new(); steps + 1];
    for (t_spike, weight) in presyn_spikes {
        let idx = (t_spike / dt).round() as isize;
        if idx >= 0 && (idx as usize) < events.len() {
            events[idx as usize].push(SynapticEvent { weight });
        }
    }

    // Simulation loop
    let mut spike_times: Vec<f64> = Vec::new();
    for step in 0..steps {
        let t = step as f64 * dt;

        // deliver presynaptic events scheduled for this time step
        for ev in &events[step] {
            neuron.receive_spike(ev.weight);
        }

        // step the neuron
        let res = neuron.step(dt, i_ext);

        if res.spiked {
            spike_times.push(t);
            // For debugging: print spike time
            println!("Spike at {:.3} ms (v reset = {:.3})", t, neuron.v);
        }

        // optionally, record v, w, s for analysis (omitted here for brevity)
        let _v = neuron.membrane_potential();
        let _w = neuron.adaptation();
        let _s = neuron.synaptic_state();

        // small example of printing membrane potential every 50 ms
        if step % ((50.0 / dt) as usize) == 0 {
            println!("t={:.1} ms, v={:.3} mV, w={:.3}, s={:.3}", t, _v, _w, _s);
        }
    }

    println!("Total spikes: {}", spike_times.len());
    println!("Spike times: {:?}", spike_times);
}
