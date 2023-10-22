/*
    Appellation: binary <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neurons::activate::Activator;

pub struct Heavyside;

impl<T> Activator<T> for Heavyside
where
    T: num::Float,
{
    fn rho(x: T) -> T {
        if x > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}
