/*
    Appellation: attrs <module>
    Created At: 2025.12.07:11:51:04
    Contrib: @FL03
*/
pub use self::{config::ConfigAttr, params::ParamsAttr};

mod config;
mod params;

/// custom attributes for the concision derive macro
#[derive(Clone, Debug, Default)]
pub struct CncAttr {
    pub config: Option<ConfigAttr>,
    pub params: Option<ParamsAttr>,
}
