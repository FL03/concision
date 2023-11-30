/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Gradient Descent
pub use self::{descent::*, gradient::*, utils::*};

pub(crate) mod descent;
pub(crate) mod gradient;

pub mod sgd;

pub struct BatchParams {
    pub batch_size: usize,
}

pub struct DescentParams {
    pub batch_size: usize,
    pub epochs: usize,
    pub gamma: f64, // learning rate

    pub lambda: f64, // decay rate
    pub mu: f64,     // momentum rate
    pub nesterov: bool,
    pub tau: f64, // momentum damper
}

pub(crate) mod utils {
    use crate::neural::prelude::{Forward, Parameterized, Params};
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
    use ndarray_stats::DeviationExt;
    use num::{FromPrimitive, Signed};

    pub fn gradient<T, D, A>(
        gamma: T,
        model: &mut A,
        data: &Array2<T>,
        targets: &Array<T, D>,
        grad: impl Fn(&Array<T, D>) -> Array<T, D>,
    ) -> f64
    where
        A: Forward<Array2<T>, Output = Array<T, D>> + Parameterized<T, D>,
        D: Dimension,
        T: FromPrimitive + NdFloat + Signed,
        <A as Parameterized<T, D>>::Params: Params<T, D> + 'static,
        Array2<T>: Dot<Array<T, D>, Output = Array<T, D>>,
    {
        let (_samples, _inputs) = data.dim();
        let pred = model.forward(data);

        let ns = T::from(data.len()).unwrap();

        let errors = &pred - targets;
        // compute the gradient of the objective function w.r.t. the model's weights
        let dz = &errors * grad(&pred);
        // compute the gradient of the objective function w.r.t. the model's weights
        let dw = data.t().to_owned().dot(&dz) / ns;
        // compute the gradient of the objective function w.r.t. the model's bias
        // let db = dz.sum_axis(Axis(0)) / ns;
        // // Apply the gradients to the model's learnable parameters
        // model.params_mut().bias_mut().scaled_add(-gamma, &db.t());

        model.params_mut().weights_mut().scaled_add(-gamma, &dw.t());

        let loss = targets
            .mean_sq_err(&model.forward(data))
            .expect("Error when calculating the MSE of the model");
        loss
    }

    pub fn gradient_descent<T, D>(
        params: &mut Array<T, D>,
        epochs: usize,
        gamma: T,
        partial: impl Fn(&Array<T, D>) -> Array<T, D>,
    ) -> Array1<T>
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
    {
        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let grad = partial(params);
            params.scaled_add(-gamma, &grad);
            losses[e] = params.mean().unwrap_or_else(T::zero);
        }
        losses
    }

    // pub fn gradient_descent_step<T, A>(
    //     args: &Array2<T>,
    //     layer: &mut Layer<T, A>,
    //     gamma: T,
    //     partial: impl Fn(&Array2<T>) -> Array2<T>,
    // ) -> T where A: Activate<Array2<T>>, T: FromPrimitive + NdFloat {
    //     let grad = partial(args);
    //     layer.weights_mut().scaled_add(-gamma, &grad);
    // }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::func::activate::{Linear, Objective, Sigmoid};
    use crate::neural::prelude::{Features, Layer, LayerShape, Parameterized, Weighted};
    use ndarray::prelude::{Array1, Array2};

    fn test_grad(args: &Array2<f64>) -> Array2<f64> {
        args.clone()
    }

    #[test]
    fn descent() {
        let (_samples, inputs, outputs) = (20, 5, 1);

        let (epochs, gamma) = (10, 0.001);

        let features = LayerShape::new(inputs, outputs);

        let mut model = Layer::<f64, Linear>::from(features).init(true);

        let losses = gradient_descent(
            &mut model.params_mut().weights_mut(),
            epochs,
            gamma,
            test_grad,
        );
        assert_eq!(losses.len(), epochs);
    }

    #[test]
    fn test_gradient() {
        let (samples, inputs, outputs) = (20, 5, 1);

        let (epochs, gamma) = (10, 0.001);

        let features = LayerShape::new(inputs, outputs);

        // Generate some example data
        let x = linarr((samples, features.inputs())).unwrap();
        let y = linarr((samples, features.outputs())).unwrap();

        let mut model = Layer::<f64, Linear>::input(features).init(true);

        let mut losses = Array1::zeros(epochs);
        for e in 0..epochs {
            let cost = gradient(gamma, &mut model, &x, &y, |w| Sigmoid::new().gradient(w));
            losses[e] = cost;
        }
        assert_eq!(losses.len(), epochs);
        assert!(losses.first() > losses.last());
    }
}
