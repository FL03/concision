/*
    Appellation: attrs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use syn::Ident;

pub struct ScsysAttr {
    pub params: Option<ParamsAttr>,
}

#[derive(Clone, Debug, Default)]
pub struct ParamsAttr {
    pub name: Option<Ident>,
}
