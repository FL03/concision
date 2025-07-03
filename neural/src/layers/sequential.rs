/*
    appellation: sequential <module>
    authors: @FL03
*/
use cnc::params::ParamsBase;
use ndarray::{Dimension, RawData};

pub struct Sequential<S, D>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) layers: Vec<ParamsBase<S, D>>,
}
