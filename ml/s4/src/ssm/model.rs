/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use ndarray::prelude::Array2;
use num::Float;


pub struct SSM<T = f64> {
    config: SSMConfig,
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
    pub d: Array2<T>,
}

impl<T> SSM<T> where T: Float {
    pub fn create(config: SSMConfig) -> Self {
        let features = config.features();
        let a = Array2::<T>::zeros(features);
        let b = Array2::<T>::zeros((features, 1));
        let c = Array2::<T>::zeros((1, features));
        let d = Array2::<T>::zeros((1, 1));
        Self { config, a, b, c, d }
    }
}
