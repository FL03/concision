/*
    Appellation: loss <module>
    Contrib: @FL03
*/

pub trait CrossEntropy {
    type Output;

    fn cross_entropy(&self) -> Self::Output;
}

pub trait MeanAbsoluteError {
    type Output;

    fn mae(&self) -> Self::Output;
}

pub trait MeanSquaredError {
    type Output;

    fn mse(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
