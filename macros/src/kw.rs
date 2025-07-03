/*
    appellation: kw <module>
    authors: @FL03
*/
//! this module defines various custom _keywords_ for the `cnc` procedural macros. These
//! keywords are used to parse the model DSL and provide a more structured way to define
//! models, layers, and their configurations.
use syn::custom_keyword;

custom_keyword! { model }
custom_keyword! { layer }
custom_keyword! { layers }
custom_keyword! { param }
custom_keyword! { params }

//
custom_keyword! { bias }
custom_keyword! { weight }

// configuration related
custom_keyword! { context }
custom_keyword! { config }
custom_keyword! { hyperparams }

// dimensional / layout related
custom_keyword! { dim }
custom_keyword! { features }
custom_keyword! { layout }
custom_keyword! { rank }
custom_keyword! { shape }
