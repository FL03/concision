#[cfg(test)]
extern crate concision_core;
extern crate concision_s4;

use concision_core as core;
use concision_s4 as s4;

use ndarray::prelude::*;
use ndarray_linalg::flatten;
use num::complex::ComplexFloat;

use core::prelude::{AsComplex, Conjugate, Power,};
use s4::cmp::kernel::{kernel_dplr, DPLRParams};
use s4::hippo::dplr::DPLR;
use s4::ops::{discrete, k_conv};

const RNGKEY: u64 = 1;



#[test]
// #[ignore = "TODO: fix this test"]
fn test_gen_dplr() {
    let (features, samples) = (4, 16);

    let eye = Array2::<f64>::eye(features);

    let step = (samples as f64).recip();

    let dplr = DPLR::<f64>::new(features);
    let (lambda, p, b, _v) = dplr.into();

    println!("{:?}", &p);


    let b2 = b.clone().insert_axis(Axis(1));

    let p2 = p.clone().insert_axis(Axis(1));

    let a = Array::from_diag(&lambda) - p2.dot(&p2.conj().t());

    // let c = {
    //     let tmp = seeded_uniform(RNGKEY, 0.0, 1.0, (1, features));
    //     println!("C:\n\n{:#?}\n", &tmp);
    //     tmp.mapv(AsComplex::as_re)
    // };
    let c = {
        let tmp = array![[0.02185547, 0.20907068, 0.23742378, 0.3723395]];
        println!("C:\n\n{:#?}\n", &tmp);
        tmp.mapv(AsComplex::as_re)
    };

    // TODO: figure out why several of the signs are wrong
    let discrete = {
        let tmp = discrete(&a, &b2, &c, step);
        assert!(tmp.is_ok(), "discretize failed: {:?}", tmp.err().unwrap());
        tmp.unwrap()
    };

    let (ab, bb, cb) = discrete.into();
    //
    let ak = k_conv(&ab, &bb, &cb.conj(), samples);
    //
    let cc = (&eye - ab.pow(samples)).conj().t().dot(&flatten(cb));
    //
    let params = DPLRParams::new(
        lambda,
        p.clone(),
        p.clone(),
        b.clone(),
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

