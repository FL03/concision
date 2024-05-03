/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::Decrement;
use nd::*;

pub(crate) fn build_bias<S, D, F>(
    biased: bool,
    dim: D,
    builder: F,
) -> Option<ArrayBase<S, D::Smaller>>
where
    S: RawData,
    D: RemoveAxis,
    F: Fn(D::Smaller) -> ArrayBase<S, D::Smaller>,
{
    if biased {
        Some(builder(dim.dec()))
    } else {
        None
    }
}
