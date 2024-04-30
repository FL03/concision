/*
    Appellation: module <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Biased, ModuleParams, Weighted};
use concision::traits::Forward;
use ndarray::prelude::{Array2, NdFloat};
use num::Float;

pub struct LinearModel<T = f64> {
    params: ModuleParams<String, T>,
}

impl<T> LinearModel<T>
where
    T: Float,
{
    pub fn new() -> Self {
        Self {
            params: ModuleParams::new(),
        }
    }

    pub fn biased(&self) -> bool {
        self.params.contains_key("bias")
    }

    pub fn weighted(&self) -> bool {
        self.params.contains_key("weight")
    }

    pub fn with_weight(mut self, weight: Array2<T>) -> Self {
        self.params.insert("weight".to_string(), weight);
        self
    }

    pub fn with_bias(mut self, bias: Array2<T>) -> Self {
        self.params.insert("bias".to_string(), bias);
        self
    }
}

// impl<T> Module<Array2<T>> for LinearModel<Array2<T>>
// where
//     T: NdFloat,
// {
//     fn name(&self) -> &str {
//         "LinearModel"
//     }

//     fn parameters(&self) -> &ModuleParams<String, T> {
//         &self.params
//     }

//     fn parameters_mut(&mut self) -> &mut ModuleParams<String, T> {
//         &mut self.params
//     }
// }

impl<T> Biased<T> for LinearModel<T>
where
    T: Float,
{
    type Dim = ndarray::Ix2;

    fn bias(&self) -> &Array2<T> {
        &self.params["bias"]
    }

    fn bias_mut(&mut self) -> &mut Array2<T> {
        self.params.get_mut("bias").unwrap()
    }

    fn set_bias(&mut self, bias: Array2<T>) {
        self.params.insert("bias".to_string(), bias);
    }
}

impl<T> Weighted<T> for LinearModel<T>
where
    T: Float,
{
    type Dim = ndarray::Ix2;

    fn weights(&self) -> &Array2<T> {
        &self.params["weight"]
    }

    fn weights_mut(&mut self) -> &mut Array2<T> {
        self.params.get_mut("weight").unwrap()
    }

    fn set_weights(&mut self, weights: Array2<T>) {
        self.params.insert("weight".to_string(), weights);
    }
}

impl<T> Forward<Array2<T>> for LinearModel<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Array2<T> {
        if let Some(weight) = self.params.get("weight") {
            if let Some(bias) = self.params.get("bias") {
                return args.dot(&weight.t()) + bias;
            }
            return args.dot(&weight.t());
        }
        args.clone()
    }
}
