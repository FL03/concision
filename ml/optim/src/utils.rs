/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;
use ndarray_stats::DeviationExt;

pub fn minimize_inner<FG>(w: &mut Array1<f64>, fg: FG, epsilon: f64) -> anyhow::Result<(&mut Array1<f64>, f64, Array1<f64>)>
where
    FG: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{

    let (mut fp, mut gp) = fg(&w);

    loop {
        w.scaled_add(-epsilon, &gp);
        let (f, g) = fg(&w);

        let expected_decrease = epsilon * norm_l2(&g);
        let actual_decrease = fp - f;
        if actual_decrease < expected_decrease * 0.5 {
            return Err(anyhow::anyhow!("Not enough decrease")); 
        }
        if actual_decrease < 2.220446049250313e-09 {
            return Ok((w, f, g));
        }
        fp = f;
        gp = g;
    }
}

fn norm_l2(a_s: &Array1<f64>) -> f64 {
    a_s.fold(0.0, |b, a| b + a * a)
}