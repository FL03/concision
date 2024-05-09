/*
   Appellation: train <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait ApplyGradient {
    type Gradient;

    fn apply_gradient(&mut self, gradient: &Self::Gradient);
}

/// [Backward] describes an object capable of backward propagation.
pub trait Backward {
    type Output;

    fn backward(&self) -> Self::Output;
}

pub trait Compile {
    type Dataset;

    fn compile(&mut self, dataset: &Self::Dataset);
}

pub trait Train: Compile {
    type Output;

    fn train(&mut self) -> Self::Output;
}

impl<S> Backward for Option<S>
where
    S: Backward,
{
    type Output = Option<S::Output>;

    fn backward(&self) -> Self::Output {
        match self {
            Some(s) => Some(s.backward()),
            None => None,
        }
    }
}
