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
    use crate::core::prelude::{assert_atol, randc_normal};
    use num::complex::ComplexFloat;

    use crate::cmp::kernel::{kernel_dplr, DPLRParams};
    use crate::hippo::dplr::DPLR;

    const FEATURES: usize = 8;
    const RNGKEY: u64 = 1;
    const SAMPLES: usize = 16;


    #[test]
    #[ignore = "TODO: fix this test"]
    fn test_conversion() {

        let step = (SAMPLES as f64).recip();

        let dplr = DPLR::<f64>::new(FEATURES);
        let (lambda, p, b, _) = dplr.clone().into();

        // let c = randcomplex(features);
        let c = randc_normal(RNGKEY, FEATURES);

        let kernel = {
            let params = DPLRParams::new(lambda.clone(), p.clone(), p.clone(), b.clone(), c.clone());
            kernel_dplr::<f64>(&params, step, SAMPLES)
        };

        let discrete = discretize_dplr(&lambda, &p, &p, &b, &c, step, SAMPLES).expect("");
        let (ab, bb, cb) = discrete.into();

        let k2 = k_conv(&ab, &bb, &cb, SAMPLES);
        let k2r = k2.mapv(|i| i.re());

        assert_atol(&kernel, &k2r, 1e-4);
    }

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
