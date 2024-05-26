/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

/// A `Params` object is used to store the various learnable parameters of a model.
///
/// ### Specifications
///
/// - `Elem`: The type of the elements being stored
///
pub trait Params {
    type Elem;
}

pub trait ParamFeatures {
    type Dim: nd::Dimension;
}

pub trait Parameter {
    type Kind: 'static;
}
