/*
    Appellation: space <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Subspace: VectorSpace {

}

pub struct Space<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

pub trait VectorSpace {
    type Dim;

}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize, y: usize| x + y;
        let actual = f(4, 4);
        let expected: usize = 8;
        assert_eq!(actual, expected)
    }
}
