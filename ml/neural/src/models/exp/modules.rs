/*
    Appellation: modules <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Model
//!
use crate::prelude::{Forward, Weighted};
use ndarray::prelude::{Array2, NdFloat};
use num::Float;
use std::collections::HashMap;

pub type ModuleParams<K, V> = HashMap<K, Array2<V>>;

pub struct M<T: Float = f64>(Box<dyn Module<T, Output = Array2<T>>>);

pub trait Module<T = f64>: Forward<Array2<T>>
where
    T: Float,
{
    fn get_param(&self, name: &str) -> Option<&Array2<T>> {
        self.parameters().get(name)
    }

    fn name(&self) -> &str;

    fn parameters(&self) -> &ModuleParams<String, T>;

    fn parameters_mut(&mut self) -> &mut ModuleParams<String, T>;
}

pub trait ModuleExt<T = f64>: Module<T>
where
    T: Float,
{
}

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

impl<T> Module<T> for LinearModel<T>
where
    T: NdFloat,
{
    fn name(&self) -> &str {
        "LinearModel"
    }

    fn parameters(&self) -> &ModuleParams<String, T> {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut ModuleParams<String, T> {
        &mut self.params
    }
}

impl<T> Weighted<T> for LinearModel<T>
where
    T: NdFloat,
{
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
