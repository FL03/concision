/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::Float;

pub trait GraphTask<T = f64>
where
    T: Float,
{
    type G;

    fn depth(&self) -> usize {
        self.layers().len()
    }

    fn layers(&self) -> &[Self::G];
}
