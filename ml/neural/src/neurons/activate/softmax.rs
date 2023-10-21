/*
    Appellation: softmax <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;

pub fn softmax<T>(args: Array1<T>) -> Array1<T>
where
    T: num::Float,
{
    let mut res = Array1::zeros(args.len());
    let denom = args.mapv(|x| x.exp()).sum();
    for (i, x) in args.iter().enumerate() {
        res[i] = x.exp() / denom;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;

    #[test]
    fn test_softmax() {
        let args = Array1::from(vec![1.0, 2.0, 3.0]);
        let res = softmax(args).mapv(|i| i.round_to(8));
        assert_eq!(res, Array1::from(vec![0.09003057, 0.24472847, 0.66524096]));
    }
}
