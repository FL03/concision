/*
    appellation: sequential <module>
    authors: @FL03
*/
use concision_params::ParamsBase;
use ndarray::{Dimension, RawData};

#[allow(dead_code)]
pub struct Sequential<S, D>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) layers: Vec<ParamsBase<S, D>>,
}
