#[cfg(test)]
extern crate concision_core;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray_linalg::flatten;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::rand_distr::uniform::{SampleUniform, Uniform};
use num::complex::ComplexFloat;

use core::prelude::{AsComplex, Conjugate, GenerateRandom, Power};
use s4::cmp::kernel::{kernel_dplr, DPLRParams};
use s4::hippo::dplr::DPLR;
use s4::ops::{discretize, k_conv};

const RNGKEY: u64 = 1;

fn seeded_uniform<T, D>(key: u64, start: T, stop: T, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: SampleUniform,
{
    Array::random_using(shape, Uniform::new(start, stop), &mut StdRng::seed_from_u64(key))
}

fn seeded_stdnorm<T, D>(key: u64, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random_using(shape, StandardNormal, &mut StdRng::seed_from_u64(key))
}

#[test]
fn test_gen_dplr() {
    let (features, samples) = (4, 16);

    let eye = Array2::<f64>::eye(features);

    let step = (samples as f64).recip();

    let dplr = DPLR::<f64>::new(features);

    let lambda = dplr.lambda.clone();

    let b2 = dplr.b.clone().insert_axis(Axis(1));

    let p2 = dplr.p.clone().insert_axis(Axis(1));

    let a = Array::from_diag(&lambda) - p2.dot(&p2.conj().t());

    let c = seeded_stdnorm(RNGKEY, features);
    let c2 = c.clone().insert_axis(Axis(0)).mapv(AsComplex::as_re);

    let discrete = {
        let tmp = discretize(&a, &b2, &c2, step.as_re());
        assert!(tmp.is_ok(), "discretize failed: {:?}", tmp.err().unwrap());
        tmp.unwrap()
    };

    let (ab, bb, cb) = discrete.into();
    //
    let ak = k_conv(&ab, &bb, &cb.conj(), samples);
    //
    let cc = {
        let tmp = flatten(cb);
        (&eye - ab.pow(samples)).conj().t().dot(&tmp)
    };
    //
    let params = DPLRParams::new(
        lambda,
        dplr.p.clone(),
        dplr.p.clone(),
        dplr.b.clone(),
        cc,
    );
    //
    let kernal = kernel_dplr::<f64>(&params, step, samples);
    println!("Kernal: {:?}", kernal.shape());

    let a_real = ak.mapv(|i| i.re());
    let err = (&a_real - &kernal).mapv(|i| i.abs());
    assert!(
        err.mean().unwrap() <= 1e-4,
        "Error: {:?}\nTolerance: {:?}",
        err.mean().unwrap(),
        1e-4
    );
}

