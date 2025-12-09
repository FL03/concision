/*
    Appellation: fft <module>
    Contrib: @FL03
*/
use cnc::Forward;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rustfft::num_complex::Complex;
use rustfft::{FftNum, FftPlanner};

/// FFT-based attention mechanism for temporal pattern recognition.
///
/// This implementation is based on "The FFT Strikes Back: Fast and Accurate
/// Spectral-Pruning Neural Networks" (https://arxiv.org/pdf/2502.18394).
///
/// The mechanism:
///
/// 1. Transforms input to frequency domain using FFT
/// 2. Applies soft thresholding to frequency components based on energy distribution
/// 3. Enhances important frequency patterns
/// 4. Returns to time domain with inverse FFT
///
/// The attention mechanism is parameterized by `steepness` and `threshold`, which control the
/// sensitivity of the attention to frequency components.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FftAttention<A = f32> {
    pub(crate) steepness: A,
    pub(crate) threshold: A,
}

impl<A> FftAttention<A> {
    /// Create a new attention module with the given parameters
    pub fn new() -> Self
    where
        A: FromPrimitive,
    {
        Self {
            steepness: A::from_f32(10.0).unwrap(),
            threshold: A::from_f32(0.1).unwrap(),
        }
    }
    /// returns an immutable reference to the steepness of the attention module
    pub const fn steepness(&self) -> &A {
        &self.steepness
    }
    /// returns a mutable reference to the steepness of the attention module to allow for
    /// gradient descent
    #[inline]
    pub fn steepness_mut(&mut self) -> &mut A {
        &mut self.steepness
    }
    /// returns an immutable reference to the threshold of the attention module
    pub const fn threshold(&self) -> &A {
        &self.threshold
    }
    /// returns a mutable reference to the threshold of the attention module to allow for
    /// gradient descent
    #[inline]
    pub fn threshold_mut(&mut self) -> &mut A {
        &mut self.threshold
    }
    /// set the steepness of the attention mechanism
    #[inline]
    pub fn set_steepness(&mut self, steepness: A) {
        self.steepness = steepness;
    }
    /// set the threshold of the attention mechanism
    #[inline]
    pub fn set_threshold(&mut self, threshold: A) {
        self.threshold = threshold;
    }
    /// consumes the current instance and returns another with the given steepness
    pub fn with_steepness(self, steepness: A) -> Self {
        Self { steepness, ..self }
    }
    /// consumes the current instance and returns another with the given threshold
    pub fn with_threshold(self, threshold: A) -> Self {
        Self { threshold, ..self }
    }
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip(self, input),
            name = "forward",
            target = "attention",
            level = "trace",
        )
    )]
    pub fn forward<X, Y>(&self, input: &X) -> Y
    where
        Self: Forward<X, Output = Y>,
    {
        <Self as Forward<X>>::forward(self, input)
    }
}

impl<A> Default for FftAttention<A>
where
    A: FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, S> Forward<ArrayBase<S, Ix1>> for FftAttention<A>
where
    A: FftNum + Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
{
    type Output = Array1<A>;

    fn forward(&self, input: &ArrayBase<S, Ix1>) -> Self::Output {
        let seq_len = input.dim();
        let n = A::from_usize(seq_len).unwrap();

        if seq_len == 0 {
            return Err(cnc::params::ParamsError::MismatchedShapes {
                expected: &[1],
                found: 0,
            }
            .into());
        }

        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(seq_len);

        // Simplified: directly use a 1D Vec for frequency domain
        let mut windowed_input: Vec<Complex<A>> = Vec::with_capacity(seq_len);

        // declare constants for windowing
        let n_minus_1 = A::from_usize(seq_len - 1).unwrap();
        let pi2 = A::from_f32(2.0 * core::f32::consts::PI).unwrap();

        // Apply windowing function while extracting data
        for time_idx in 0..seq_len {
            // Apply Hann window: 0.5 * (1 - cos(2π*i/(N-1)))
            let window_factor = if seq_len > 1 {
                let i_f = A::from_usize(time_idx).unwrap();
                A::from_f32(0.5).unwrap() * (A::one() - (pi2 * i_f / n_minus_1).cos())
            } else {
                A::one()
            };

            // Get value and apply window
            let val = input[time_idx] * window_factor;
            windowed_input.push(Complex::new(val, A::zero()));
        }

        // Perform FFT in-place
        fft.process(&mut windowed_input);

        // Calculate energy and total energy directly
        let mut freq_energy = Array1::<A>::zeros(seq_len);
        let mut total_energy = A::zero();
        for (time_idx, &val) in windowed_input.iter().enumerate() {
            let energy = (val.re * val.re + val.im * val.im).sqrt();
            freq_energy[time_idx] = energy;
            total_energy = total_energy + energy;
        }

        // Add epsilon to prevent division by zero
        let epsilon = A::from_f32(1e-10).unwrap();
        total_energy = total_energy.max(epsilon);

        // Clip normalized energy values to prevent sigmoid explosion
        for time_idx in 0..seq_len {
            // normalize energy
            let normalized_energy = freq_energy[time_idx] / total_energy;

            // Use a more stable sigmoid implementation
            let exp_term = (-(normalized_energy - self.threshold) * self.steepness).exp();
            let attention_weight = if exp_term.is_finite() {
                A::one() / (A::one() + exp_term)
            } else if (normalized_energy - self.threshold) > A::zero() {
                A::one() // Sigmoid approaches 1 for large positive inputs
            } else {
                A::zero() // Sigmoid approaches 0 for large negative inputs
            };

            // Apply weight
            windowed_input[time_idx] = Complex::new(
                windowed_input[time_idx].re * attention_weight,
                windowed_input[time_idx].im * attention_weight,
            );
        }

        // Inverse FFT in-place
        let ifft = planner.plan_fft_inverse(seq_len);
        ifft.process(&mut windowed_input);

        // Create a result array with same dimensions as input
        let mut result = Array1::zeros(seq_len);
        if windowed_input
            .iter()
            .any(|&c| c.re.is_nan() || c.im.is_nan())
        {
            #[cfg(feature = "tracing")]
            tracing::warn!("The FFT/IFFT process produced NaN values.");
        }
        // Transfer back the processed values from frequency domain
        for (idx, &complex) in windowed_input.iter().enumerate() {
            // Normalize by sequence length (standard for IFFT)
            let res = complex.re / n;
            if res.is_nan() {
                result[idx] = A::zero(); // Replace NaN with zero
            } else {
                result[idx] = res;
            }
        }

        Ok(result)
    }
}

impl<A, S> Forward<ArrayBase<S, Ix2>> for FftAttention<A>
where
    A: FftNum + Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn forward(&self, input: &ArrayBase<S, Ix2>) -> cnc::Result<Self::Output> {
        use rustfft::FftPlanner;
        use rustfft::num_complex::Complex;

        let (seq_len, feature_dim) = input.dim();

        if seq_len == 0 {
            return Err(anyhow::anyhow!("Input sequence length cannot be zero"));
        }

        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(seq_len);
        let mut frequency_domain = Array2::<Complex<A>>::zeros((feature_dim, seq_len));

        // declare constants for windowing
        let n_minus_1 = A::from_usize(seq_len - 1).unwrap();
        let pi2 = A::from_f32(2.0 * core::f32::consts::PI).unwrap();
        // Process each feature dimension
        for feature_idx in 0..feature_dim {
            // Extract this feature across all timesteps
            let mut windowed_input: Vec<Complex<A>> = Vec::with_capacity(seq_len);

            // Apply windowing function while extracting data
            for time_idx in 0..seq_len {
                // Apply Hann window: 0.5 * (1 - cos(2π*i/(N-1)))
                let window_factor = if seq_len > 1 {
                    let i_f = A::from_usize(time_idx).unwrap();
                    A::from_f32(0.5).unwrap() * (A::one() - (pi2 * i_f / n_minus_1).cos())
                } else {
                    A::one()
                };

                // Get value and apply window
                let val = input[[time_idx, feature_idx]] * window_factor;
                windowed_input.push(Complex::new(val, A::zero()));
            }

            // Perform FFT
            fft.process(&mut windowed_input);

            // Store in frequency domain
            for (time_idx, &val) in windowed_input.iter().enumerate() {
                frequency_domain[[feature_idx, time_idx]] = val;
            }
        }

        // Calculate frequency domain attention weights
        let mut attention_weights = Array2::<A>::zeros((feature_dim, seq_len));

        for fdx in 0..feature_dim {
            // Calculate energy at each frequency
            let mut total_energy = A::zero();
            let mut freq_energy = Array1::<A>::zeros(seq_len);

            for time_idx in 0..seq_len {
                let val = frequency_domain[[fdx, time_idx]];
                let energy = (val.re * val.re + val.im * val.im).sqrt();
                freq_energy[time_idx] = energy;
                total_energy = total_energy + energy;
            }

            // Normalize to create attention weights
            if total_energy.is_positive() {
                for time_idx in 0..seq_len {
                    // Apply soft-thresholding as described in paper
                    let normalized_energy = freq_energy[time_idx] / total_energy;

                    // Using sigmoid to create attention weight with soft threshold
                    attention_weights[[fdx, time_idx]] = A::one()
                        / (A::one()
                            + (-(normalized_energy - self.threshold) * self.steepness).exp());
                }
            }
        }

        // Apply attention weights in frequency domain
        frequency_domain
            .iter_mut()
            .zip(attention_weights.iter())
            .for_each(|(val, &w)| {
                *val = Complex::new(val.re * w, val.im * w);
            });

        // Create a result array with same dimensions as input
        let mut result = Array2::zeros((seq_len, feature_dim));

        // Inverse FFT to get back to time domain with enhanced patterns
        let ifft = planner.plan_fft_inverse(seq_len);

        for fdx in 0..feature_dim {
            let mut row = frequency_domain.row_mut(fdx);
            let mut feature_slice = row.as_slice_mut().unwrap();

            // Perform inverse FFT
            ifft.process(&mut feature_slice);

            // Transfer back to result array, preserving time dimension
            for (t_idx, complex) in feature_slice.iter().enumerate() {
                // Normalize by sequence length (standard for IFFT)
                result[[t_idx, fdx]] = complex.re / A::from_f32(seq_len as f32).unwrap();
            }
        }

        Ok(result)
    }
}
