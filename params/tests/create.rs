/*
    Appellation: params <test>
    Contrib: @FL03
*/
use concision_params::Params;

#[test]
fn test_params_ones() {
    // weights retain the given shape (in, out)
    // bias retains the shape (out,)
    let ones = Params::<f64>::ones((3, 4));
    assert_eq!(ones.dim(), (3, 4));
    assert_eq!(ones.bias().dim(), 4);
    assert!(
        ones.iter()
            .all(|(w, b)| w.iter().all(|&wi| wi == 1.0) && b == &1.0)
    );
}

#[test]
fn test_params_zeros() {
    // weights retain the given shape (in, out)
    // bias retains the shape (out,)
    let zeros = Params::<f64>::zeros((3, 4));
    assert_eq!(zeros.dim(), (3, 4));
    assert_eq!(zeros.bias().dim(), 4);
    assert!(
        zeros
            .iter()
            .all(|(w, b)| w.iter().all(|&wi| wi == 0.0) && b == &0.0)
    );
}

#[test]
#[cfg(feature = "rand")]
fn test_params_init_rand() -> anyhow::Result<()> {
    use concision_init::RandTensor;

    let lecun = Params::<f64>::lecun_normal((3, 4));
    assert_eq!(lecun.dim(), (3, 4));

    let glorot_norm = Params::<f64>::glorot_normal((3, 4));
    assert_eq!(glorot_norm.dim(), (3, 4));
    assert_ne!(lecun, glorot_norm);
    let glorot_uniform = Params::<f64>::glorot_uniform((3, 4)).expect("glorot_uniform failed");
    assert_eq!(glorot_uniform.dim(), (3, 4));
    assert_ne!(lecun, glorot_uniform);
    assert_ne!(glorot_norm, glorot_uniform);
    let truncnorm = Params::<f64>::truncnorm((3, 4), 0.0, 1.0).expect("truncnorm failed");
    assert_eq!(truncnorm.dim(), (3, 4));

    Ok(())
}
