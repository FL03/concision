/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, Dimension, RawData};

/// [Parameters] describes any object capable of serving as a store for the
/// model's learnable parameters.
///
/// ### Specifications
///
/// - `Elem`: The type of the elements being stored
///
pub trait Parameters {
    type Elem;
}

pub trait ParamFeatures {
    type Dim: nd::Dimension;
}

/// A `Parameter` describes learnable parameters in a model.
pub trait Parameter<T> {
    type Data;
}

pub trait ParameterExt<T>: Parameter<T> {
    type Kind: 'static;
}

/*
 ************* Implementations *************
*/

impl<A, S, D> Parameter<A> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Data = S;
}

impl<A, S, D> Parameters for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Elem = A;
}
