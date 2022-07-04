/*
   Appellation: matrices
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Matrix<DType = usize> {
    pub data: Vec<Vec<DType>>
}

pub trait Matrices<Data: Sized = f32>: Sized {

    fn constructor(args: Vec<Data>) -> Self where Self: Sized;

}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
