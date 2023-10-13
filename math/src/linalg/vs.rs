/*
    Appellation: vs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VectorSpace<T>(pub Vec<T>);

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
