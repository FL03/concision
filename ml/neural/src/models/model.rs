/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{ModelConfig, ModelParams};
use crate::prelude::{Forward, LayerParams};
use ndarray::prelude::{Array2, NdFloat};
use num::Float;

#[derive(Clone, Debug)]
pub struct Model<T = f64>
where
    T: Float,
{
    config: ModelConfig,
    params: ModelParams<T>,
}

impl<T> Model<T>
where
    T: Float,
{
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            params: ModelParams::new(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ModelConfig {
        &mut self.config
    }

    pub fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}

// impl<T> FromIterator<LayerShape> for Model<T>
// where
//     T: Float,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = LayerShape>,
//     {
//         let params = ModelParam::from_iter(iter);
//         Self {

//         }
//     }
// }

// impl<T> FromIterator<LayerParams<T>> for Model<T>
// where
//     T: Float,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = LayerParams<T>>,
//     {
//         Self {
//             children: iter.into_iter().collect(),
//         }
//     }
// }
impl<T> IntoIterator for Model<T>
where
    T: Float,
{
    type Item = LayerParams<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<T> Forward<Array2<T>> for Model<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        let mut iter = self.clone().into_iter();

        let mut output = iter.next().unwrap().forward(input);
        for layer in iter {
            output = layer.forward(&output);
        }
        output
    }
}
