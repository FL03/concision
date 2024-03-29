/*
    Appellation: adam <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Adam (Adaptive Moment Estimation) optimizer
//!
//! The `Adam (Adaptive Moment Estimation)` optimizer is an adaptive learning rate algorithm used
//! in gradient descent and machine learning, such as for training neural networks to solve deep
//! learning problems. Boasting memory-efficient fast convergence rates, it sets and iteratively
//! updates learning rates individually for each model parameter based on the gradient history.
//!
//! ## Algorithm:
//!
//! Given:
//!   - α is the learning rate
//!   - (β_1, β_2) are the exponential decay rates for moment estimates
//!   - ϵ is any small value to prevent division by zero
//!   - g_t are the gradients at time step t
//!   - m_t are the biased first moment estimates of the gradient at time step t
//!   - v_t are the biased second raw moment estimates of the gradient at time step t
//!   - θ_t are the model parameters at time step t
//!   - t is the time step
//!
//! Required:
//!   θ_0
//!
//! Initialize:
//!   m_0 <- 0
//!   v_0 <- 0
//!   t <- 0
//!
//! while θ_t not converged do
//!   m_t = β_1 * m_{t−1} + (1 − β_1) * g_t
//!   v_t = β_2 * v_{t−1} + (1 − β_2) * g_t^2
//!   m_hat_t = m_t / 1 - β_1^t
//!   v_hat_t = v_t / 1 - β_2^t
//!   θ_t = θ_{t-1} − α * m_hat_t / (sqrt(v_hat_t) + ϵ)
//!
//! ## Resources:
//!   - Adam: A Method for Stochastic Optimization (by Diederik P. Kingma and Jimmy Ba):
//!       - [https://arxiv.org/abs/1412.6980]
//!   - PyTorch Adam optimizer:
//!       - [https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam]
//!
use ndarray::prelude::{Array1, NdFloat};
use num::Float;

pub struct Adam<T = f64>
where
    T: Float,
{
    learning_rate: T, // alpha: initial step size for iterative optimization
    betas: (T, T),    // betas: exponential decay rates for moment estimates
    epsilon: T,       // epsilon: prevent division by zero
    m: Array1<T>,     // m: biased first moment estimate of the gradient vector
    v: Array1<T>,     // v: biased second raw moment estimate of the gradient vector
    t: usize,         // t: time step
}

impl<T> Adam<T>
where
    T: Float,
{
    pub fn new(
        learning_rate: Option<T>,
        betas: Option<(T, T)>,
        epsilon: Option<T>,
        params_len: usize,
    ) -> Self {
        let gamma = T::from(1e-3).unwrap();
        let betas = betas.unwrap_or((T::from(0.9).unwrap(), T::from(0.999).unwrap()));
        let epsilon = epsilon.unwrap_or(T::from(1e-8).unwrap());
        Adam {
            learning_rate: learning_rate.unwrap_or(gamma), // typical good default lr
            betas,                                         // typical good default decay rates
            epsilon,                                       // typical good default epsilon
            m: Array1::zeros(params_len), // first moment vector elements all initialized to zero
            v: Array1::zeros(params_len), // second moment vector elements all initialized to zero
            t: 0,                         // time step initialized to zero
        }
    }
}

impl<T> Adam<T>
where
    T: NdFloat,
{
    pub fn step(&mut self, gradients: &Array1<T>) -> Array1<T> {
        let mut model_params = Array1::zeros(gradients.len());
        self.t += 1;

        for i in 0..gradients.len() {
            // update biased first moment estimate and second raw moment estimate
            self.m[i] = self.betas.0 * self.m[i] + (T::one() - self.betas.0) * gradients[i];
            self.v[i] = self.betas.1 * self.v[i] + (T::one() - self.betas.1) * gradients[i].powi(2);

            // compute bias-corrected first moment estimate and second raw moment estimate
            let m_hat = self.m[i] / (T::one() - self.betas.0.powi(self.t as i32));
            let v_hat = self.v[i] / (T::one() - self.betas.1.powi(self.t as i32));

            // update model parameters
            model_params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        model_params // return updated model parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::array;

    #[test]
    fn test_adam_init_default_values() {
        let optimizer = Adam::new(None, None, None, 1);

        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.betas, (0.9, 0.999));
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.m, array![0.0]);
        assert_eq!(optimizer.v, array![0.0]);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_init_custom_lr_value() {
        let optimizer = Adam::new(Some(0.9), None, None, 2);

        assert_eq!(optimizer.learning_rate, 0.9);
        assert_eq!(optimizer.betas, (0.9, 0.999));
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.m, array![0.0, 0.0]);
        assert_eq!(optimizer.v, array![0.0, 0.0]);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_init_custom_betas_value() {
        let optimizer = Adam::new(None, Some((0.8, 0.899)), None, 3);

        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.betas, (0.8, 0.899));
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.m, array![0.0, 0.0, 0.0]);
        assert_eq!(optimizer.v, array![0.0, 0.0, 0.0]);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_init_custom_epsilon_value() {
        let optimizer = Adam::new(None, None, Some(1e-10), 4);

        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.betas, (0.9, 0.999));
        assert_eq!(optimizer.epsilon, 1e-10);
        assert_eq!(optimizer.m, array![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(optimizer.v, array![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_init_all_custom_values() {
        let optimizer = Adam::new(Some(1.0), Some((0.001, 0.099)), Some(1e-1), 5);

        assert_eq!(optimizer.learning_rate, 1.0);
        assert_eq!(optimizer.betas, (0.001, 0.099));
        assert_eq!(optimizer.epsilon, 1e-1);
        assert_eq!(optimizer.m, Array1::<f64>::zeros(5));
        assert_eq!(optimizer.v, Array1::<f64>::zeros(5));
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_step_default_params() {
        let gradients = array![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

        let mut optimizer = Adam::new(None, None, None, 8);
        let updated_params = optimizer.step(&gradients);

        assert_eq!(
            updated_params,
            array![
                0.0009999999900000003,
                -0.000999999995,
                0.0009999999966666666,
                -0.0009999999975,
                0.000999999998,
                -0.0009999999983333334,
                0.0009999999985714286,
                -0.00099999999875
            ]
        );
    }

    #[test]
    fn test_adam_step_custom_params() {
        let gradients = array![9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0];

        let mut optimizer = Adam::new(Some(0.005), Some((0.5, 0.599)), Some(1e-5), 9);
        let updated_params = optimizer.step(&gradients);

        assert_eq!(
            updated_params,
            array![
                -0.004999994444450618,
                0.004999993750007813,
                -0.004999992857153062,
                0.004999991666680556,
                -0.004999990000020001,
                0.004999987500031251,
                -0.004999983333388888,
                0.004999975000124999,
                -0.0049999500004999945
            ]
        );
    }

    #[test]
    fn test_adam_step_empty_gradients_array() {
        let gradients = Array1::<f64>::zeros(0);

        let mut optimizer = Adam::new(None, None, None, 0);
        let updated_params = optimizer.step(&gradients);

        assert_eq!(updated_params, Array1::<f64>::zeros(0));
    }

    #[ignore]
    #[test]
    fn test_adam_step_iteratively_until_convergence_with_default_params() {
        const CONVERGENCE_THRESHOLD: f64 = 1e-5;
        let gradients = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut optimizer = Adam::new(None, None, None, 6);

        let mut model_params = Array1::<f64>::zeros(6);
        let mut updated_params: Array1<f64> = optimizer.step(&gradients);

        while (updated_params
            .iter()
            .zip(model_params.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>())
        .iter()
        .map(|&x| x.powi(2))
        .sum::<f64>()
        .sqrt()
            > CONVERGENCE_THRESHOLD
        {
            model_params = updated_params;
            updated_params = optimizer.step(&gradients);
        }

        // assert!(updated_params < Array1::<f64>::from_elem(6, CONVERGENCE_THRESHOLD));
        assert_ne!(updated_params, model_params);
        assert_eq!(
            updated_params,
            array![
                -0.0009999999899999931,
                -0.0009999999949999929,
                -0.0009999999966666597,
                -0.0009999999974999929,
                -0.0009999999979999927,
                -0.0009999999983333263
            ]
        );
    }

    #[ignore]
    #[test]
    fn test_adam_step_iteratively_until_convergence_with_custom_params() {
        const CONVERGENCE_THRESHOLD: f64 = 1e-7;
        let gradients = array![7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0];

        let mut optimizer = Adam::new(Some(0.005), Some((0.8, 0.899)), Some(1e-5), 7);

        let mut model_params = Array1::<f64>::zeros(7);
        let mut updated_params = optimizer.step(&gradients);

        while (updated_params
            .iter()
            .zip(model_params.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>())
        .iter()
        .map(|&x| x.powi(2))
        .sum::<f64>()
        .sqrt()
            > CONVERGENCE_THRESHOLD
        {
            model_params = updated_params;
            updated_params = optimizer.step(&gradients);
        }

        // assert!(updated_params < vec![CONVERGENCE_THRESHOLD; 7]);
        assert_ne!(updated_params, model_params);
        assert_eq!(
            updated_params,
            array![
                -0.004999992857153061,
                0.004999993750007814,
                -0.0049999944444506185,
                0.004999995000005001,
                -0.004999995454549587,
                0.004999995833336807,
                -0.004999996153849113
            ]
        );
    }
}
