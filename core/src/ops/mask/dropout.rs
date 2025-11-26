/// [Dropout] randomly zeroizes elements with a given probability (`p`).
pub trait DropOut {
    type Output;

    fn dropout(&self, p: f64) -> Self::Output;
}

#[cfg(feature = "init")]
impl<A, S, D> DropOut for ndarray::ArrayBase<S, D, A>
where
    A: num_traits::Num + ndarray::ScalarOperand,
    D: ndarray::Dimension,
    S: ndarray::DataOwned<Elem = A>,
{
    type Output = ndarray::Array<A, D>;

    fn dropout(&self, p: f64) -> Self::Output {
        pub use concision_init::Initialize;
        use ndarray::Array;
        let dim = self.dim();
        // Create a mask of the same shape as the input array
        let mask: Array<bool, D> = Array::bernoulli(dim, p).expect("Failed to create mask");
        let mask = mask.mapv(|x| if x { A::zero() } else { A::one() });

        // Element-wise multiplication to apply dropout
        self.to_owned() * mask
    }
}
