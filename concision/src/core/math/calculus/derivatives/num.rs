/*
    Appellation: num <module>
    Contributors: FL03 <jo3mccain@icloud.com> (https://gitlab.com/FL03)
    Description:
        ... Summary ...
*/
use crate::math::Numerical;
use serde::{Deserialize, Serialize};
use strum::{EnumString, EnumVariantNames};

#[derive(
Clone,
Copy,
Debug,
Default,
Deserialize,
EnumString,
EnumVariantNames,
Eq,
Hash,
PartialEq,
PartialOrd,
Serialize,
)]
#[strum(serialize_all = "snake_case")]
pub enum DerivativeMode {
    Backwards,
    #[default]
    Central,
    Forwards,
}

impl DerivativeMode {
    pub fn execute<T>(&self, data: T, func: &dyn Fn(T) -> T) {
        match self {
            Self::Backwards => {}
            Self::Central => {}
            Self::Forwards => {}
        }
    }
}

pub fn central(x: Vec<f64>, f: &dyn Fn(f64) -> f64, h: f64) -> Vec<f64> {
    let mut res = Vec::new();
    for i in x {
        res.push((f(i.clone() + h) - f(i - h)) / h);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let f = |x: f64| x.powf(x);
        assert_eq!(f(2.0), 4.0)
    }
}
