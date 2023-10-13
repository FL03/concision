/*
    Appellation: deviation <module>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/

pub struct StandardDeviation {
    pub mean: f64,
    pub variance: f64,
    pub deviation: f64,
}

impl StandardDeviation {
    pub fn new(x: &[f64]) -> StandardDeviation {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let variance = x.iter().map(|&x| x * x).sum::<f64>() / x.len() as f64 - mean * mean;
        let deviation = variance.sqrt();
        StandardDeviation {
            mean,
            variance,
            deviation,
        }
    }
}
