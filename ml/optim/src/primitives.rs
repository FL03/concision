/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

mod constants {

    pub const FTOL: f64 = 2.220446049250313e-09;
    
}

mod statics {}

mod types {
    use ndarray::prelude::{Array1, Array2};

    pub type ObjectiveFn<T> = fn(&Array2<T>, &Array1<T>) -> Array1<T>;
}
