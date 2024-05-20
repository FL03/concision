/*
    Appellation: mask <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

pub struct Mask<S, D>(ArrayBase<S, D>)
where
    D: Dimension,
    S: RawData<Elem = bool>;
