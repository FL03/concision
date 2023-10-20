/*
    Appellation: utils <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub fn round<T: num::Float>(num: T, decimals: usize) -> T {
    let factor = T::from(10.0).expect("").powi(decimals as i32);
    (num * factor).round() / factor
}
