/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait RoundTo {
    fn round_to(&self, decimals: usize) -> Self;
}

impl<T> RoundTo for T
where
    T: num::Float,
{
    fn round_to(&self, decimals: usize) -> Self {
        let val = T::from(self.clone()).expect("Failed to convert to type T");
        let factor = T::from(10).expect("").powi(decimals as i32);
        (val * factor).round() / factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to() {
        let num = 1.23456789_f64;
        assert_eq!(num.round_to(2), 1.23_f64);
        assert_eq!(num.round_to(3), 1.235_f64);
    }
}
