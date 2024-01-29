/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;

pub struct S4State<T = f64> {
    cache: Array1<T>,
}
