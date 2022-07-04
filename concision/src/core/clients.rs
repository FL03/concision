/*
   Appellation: clients
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Databases {
    Abbreviated {
        prefix: String,
        address: String,
    },
    Simple {
        address: String,
    },
    Standard {
        prefix: String,
        username: String,
        password: String,
        host: String,
        port: String,
        suffix: String,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Connections {
    Api { token: String },
}

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Client {
    pub address: String,
    pub name: String,
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
