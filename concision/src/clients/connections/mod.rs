/*
   Appellation: connections
   Context: module
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       A collection of connections for developers to quickly setup and stream data;

*/

type ConfigBuilderDS = config::ConfigBuilder<config::builder::DefaultState>;

pub trait ConnectionSpec<Addr, Client, Conf, Data> {
    fn authenticate(&self, address: Addr, client: Client) -> Client
    where
        Self: Sized;
    fn client(&self, data: Data) -> Client
    where
        Self: Sized;
    fn configure(&self, configuration: Conf) -> Result<Self, config::ConfigError>
    where
        Self: Sized;
    fn deconstruct(&self) -> Vec<Data>
    where
        Self: Sized;
}
