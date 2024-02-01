/*
   Appellation: generate <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Generator {}

pub trait Genspace<T = f64> {
    fn arange(start: T, stop: T, step: T) -> Self;

    fn linspace(start: T, stop: T, n: usize) -> Self;

    fn logspace(start: T, stop: T, n: usize) -> Self;

    fn geomspace(start: T, stop: T, n: usize) -> Self;

    fn ones(n: usize) -> Self;

    fn zeros(n: usize) -> Self;
}
