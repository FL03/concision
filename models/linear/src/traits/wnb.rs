/*
    Appellation: wnb <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

pub trait WnB<S, D = Ix2> where D: Dimension, S: RawData {
    fn bias(&self) -> Option<&ArrayBase<S, D::Smaller>>;

    fn bias_mut(&mut self) -> Option<&mut ArrayBase<S, D::Smaller>>;

    fn weight(&self) -> &ArrayBase<S, D>;

    fn weight_mut(&mut self) -> &mut ArrayBase<S, D>;
}
