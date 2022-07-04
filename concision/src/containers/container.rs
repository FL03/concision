/*
   Appellation: container
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Container<Key = String, Data = String> {
    pub id: crate::Ids,
    pub hash: Vec<u8>,
    pub key: Key,
    pub data: Vec<Data>,
}

impl<Key, Data> Container<Key, Data> {
    pub fn constructor(
        id: crate::Ids,
        hash: Vec<u8>,
        key: Key,
        data: Vec<Data>,
    ) -> Self {
        Self {
            id,
            hash,
            key,
            data
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
