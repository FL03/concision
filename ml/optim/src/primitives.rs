/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

mod constants {}

mod statics {}

mod types {
    use ndarray::prelude::{Array1, Array2};

    pub type ObjectiveFn<T> = fn(&Array2<T>, &Array1<T>) -> Array1<T>;
}
