/*
    Appellation: attrs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use syn::Ident;

#[derive(Clone, Debug, Default)]
pub struct ConfigAttr {
    pub name: Option<Ident>,
    pub format: Option<Ident>,
}
