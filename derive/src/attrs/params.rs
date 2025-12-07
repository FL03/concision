/*
    Appellation: attrs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use syn::Ident;

#[derive(Clone, Debug, Default)]
pub struct ParamsAttr {
    pub name: Option<Ident>,
}
