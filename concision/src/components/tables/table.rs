/*
    Appellation: table <module>
    Creator: FL03 <jo3mccain@icloud.com>
    Description:
        ... Summary ...
*/
/// Define a table interface for creating custom stacks
pub trait TableSpec {
    fn shape(&self) -> (usize, usize)
        where
            Self: Sized;
}

/// Implement a standard table structure for immediate use
#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Table<T> {
    pub data: Vec<Vec<T>>,
    pub shape: (usize, usize),
}

impl<T> Table<T> {
    pub fn new(shape: (usize, usize)) -> Self {
        let mut data = Vec::<Vec<T>>::with_capacity(shape.0);
        for _ in 0..shape.1 {
            data.push(Vec::<T>::with_capacity(shape.1))
        }
        Self { data, shape }
    }
}

impl<T> Default for Table<T> {
    fn default() -> Self {
        Self::new((3, 3))
    }
}

impl<T> TableSpec for Table<T> {
    fn shape(&self) -> (usize, usize)
        where
            Self: Sized,
    {
        *&self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_shape() {
        let actual = Table::<f64>::default();
        let expected = Table::<f64>::new((3, 3));
        println!("{:#?}", actual.clone());
        assert_eq!(actual, expected)
    }
}
