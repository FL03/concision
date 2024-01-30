/*
   Appellation: arithmetic <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Arithmetic {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
}
