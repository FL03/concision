/*
   Appellation: container
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/
pub use specs::*;

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Container<Key = String, Data = String> {
    pub id: crate::Ids,
    pub hash: Vec<u8>,
    pub key: Key,
    pub data: Vec<Data>,
}

impl<Key, Data> Container<Key, Data> {
    pub fn constructor(id: crate::Ids, hash: Vec<u8>, key: Key, data: Vec<Data>) -> Self {
        Self {
            id,
            hash,
            key,
            data,
        }
    }
}

impl std::fmt::Display for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Container(id={:#?}, hash={:#?}, key={:#?}, data={:#?})",
            self.id, self.hash, self.key, self.data
        )
    }
}

mod specs {
    pub trait ContainerSpec<Data = Vec<String>> {
        fn constructor(&self, data: Data) -> Self
        where
            Self: Sized;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let container: Box<dyn ContainerSpec>;
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
