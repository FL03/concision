/*
    Appellation: gets <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use proc_macro2::TokenStream;

pub fn _gets(input: TokenStream) -> TokenStream {
    println!("display: {:?}", input);
    input
}
