//! # Example: Leaky Integrate-and-Fire Neuron
//!
//! This example demonstrates the usage of a Leaky Integrate-and-Fire (LIF) neuron model,
//! simulating its behavior over time with constant external current injection and discrete
//! synaptic events.
use concision_ext::snn::{Leaky, SynapticEvent};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_timer(tracing_subscriber::fmt::time::Uptime::default())
        .init();
    // Simulation parameters
    let dt: f64 = 0.1; // ms
    let t_sim: f64 = 5000.0; // ms
    let steps = (t_sim / dt) as usize;

    // Create neuron with defaults
    let mut neuron = Leaky::<f64>::default();

    // Example external current (constant)
    // Increase drive so steady-state v can reach threshold (v_rest + R*i_ext > v_thresh).
    // With default params: v_rest = -65, v_thresh = -50 → need i_ext >= 15.
    let i_ext = 15.0; // stronger constant drive to induce spiking

    // Example presynaptic spike times (ms) and weights
    let presyn_spikes: Vec<(f64, f64)> =
        vec![(50.0, 2.0), (100.0, 1.5), (150.0, 2.2), (300.0, 3.0)];

    // Convert into an index-able event list
    let mut events: Vec<Vec<SynapticEvent>> = vec![Vec::new(); steps + 1];
    for (t_spike, weight) in presyn_spikes {
        let idx = (t_spike / dt).round() as isize;
        if idx >= 0 && (idx as usize) < events.len() {
            events[idx as usize].push(SynapticEvent::new(weight));
        }
    }

    // Simulation loop
    let mut spike_times = Vec::<f64>::new();
    for step in 0..steps {
        let t = step as f64 * dt;

        // deliver presynaptic events scheduled for this time step
        for ev in &events[step] {
            neuron.apply_spike(ev.weight);
        }

        // step the neuron
        let res = neuron.step(dt, i_ext);

        if res.is_spiked() {
            spike_times.push(t);
            // print the pre-spike membrane potential from the step result
            tracing::info!("Spike at {:.3} ms (pre-spike v = {:.3})", t, res.get());
        }

        // optionally, record v, w, s for analysis (omitted here for brevity)
        let _v = neuron.membrane_potential();
        let _w = neuron.adaptation();
        let _s = neuron.synaptic_state();

        // small example of printing membrane potential every 50 ms
        if step % ((50.0 / dt) as usize) == 0 {
            // show the step result potential (pre-reset when a spike occurred)
            tracing::trace!(
                "t={:.1} ms, v={:.3} mV, w={:.3}, s={:.3}",
                t,
                res.get(),
                _w,
                _s
            );
        }
    }

    tracing::info!("Total spikes: {}", spike_times.len());
    tracing::info!("Spike times: {:?}", spike_times);

    Ok(())
}
