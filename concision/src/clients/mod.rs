/*
   Appellation: containers
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

mod connections;

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