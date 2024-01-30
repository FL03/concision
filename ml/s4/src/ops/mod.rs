/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{convolve::*, discretize::*, gen::*, scan::*};

pub(crate) mod convolve;
pub(crate) mod discretize;
pub(crate) mod gen;
pub(crate) mod scan;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::randc_normal;
    use crate::hippo::dplr::DPLR;

    const FEATURES: usize = 8;
    const RNGKEY: u64 = 1;
    const SAMPLES: usize = 16;


    #[test]
    fn test_discretize() {
        let step = (SAMPLES as f64).recip();

        let c = randc_normal(RNGKEY, FEATURES);

        let dplr = DPLR::<f64>::new(FEATURES);
        let (lambda, p, b, _) = dplr.clone().into();

        let _discrete = {
            let tmp = discretize_dplr(&lambda, &p, &p, &b, &c, step, SAMPLES);
            assert!(tmp.is_ok(), "Error: {:?}", tmp.err());
            tmp.unwrap()
        };
    }
}
