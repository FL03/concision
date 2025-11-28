/*
    Appellation: training <module>
    Created At: 2025.11.28:13:07:13
    Contrib: @FL03
*/

/// This trait defines the training process for the network
pub trait Train<X, Y> {
    type Error;
    type Output;

    fn train(&mut self, input: &X, target: &Y) -> Result<Self::Output, Self::Error>;

    fn train_for(
        &mut self,
        input: &X,
        target: &Y,
        epochs: usize,
    ) -> Result<Self::Output, Self::Error> {
        let mut output = None;
        let mut step: usize = 0;

        while step < epochs {
            step += 1;
            output = Some(self.train(input, target)?);
        }

        for _ in 0..epochs {
            output = match self.train(input, target) {
                Ok(o) => Some(o),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        if let Some(o) = output {
            Ok(o)
        } else {
            panic!("Training did not produce any output")
        }
    }
}
