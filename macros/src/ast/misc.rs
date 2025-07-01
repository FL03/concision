/*
    appellation: misc <module>
    authors: @FL03
*/
use syn::Ident;
use syn::parse::{Parse, ParseStream};
use syn::token::Colon;

#[allow(dead_code)]
/// The [`BlockHeader`] implementation provides a mechanism for _tagging_ a block of code that
/// is marked / defined by a particular keyword. The header will look for an optional
/// [`NameTag`] (i.e. a colon and an ident) before the opening brace of the block.
///
/// For example, the block header enables a macro to parse both
///
/// ```no_run
/// my_macro! {
///     keyword { ... }
/// }
/// ```
///
/// and
///
/// ```no_run
/// my_macro! {
///     keyword: Name { ... }
/// }
/// ```
///
/// where the `keyword` is a keyword that is defined in the macro, such as `config`, `params`,
/// etc.
pub struct BlockHeader<Key> {
    // the `config` keyword
    pub key: Key,
    pub nametag: Option<NameTag>,
}

#[allow(dead_code)]
/// an optional _nametag_ following the keyword of a particular block.
/// This is used to provide a more descriptive name for the block, such as `params: ___ {}`.
pub struct NameTag {
    pub colon_token: Colon,
    pub ident: Ident,
}

impl<Key> Parse for BlockHeader<Key>
where
    Key: Parse,
{
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the key
        let key = input.parse::<Key>()?;
        // optionally parse the nametag
        let nametag = if input.peek(Colon) {
            Some(input.parse::<NameTag>()?)
        } else {
            None
        };
        // return the BlockHeader instance
        Ok(BlockHeader { key, nametag })
    }
}

impl Parse for NameTag {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the colon token
        let colon_token = input.parse::<Colon>()?;
        // parse the identifier
        let ident = input.parse::<Ident>()?;
        // return the NameTag instance
        Ok(NameTag { colon_token, ident })
    }
}
