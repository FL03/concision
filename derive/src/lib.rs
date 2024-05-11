/*
   Appellation: concision-derive <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_derive"]
#![crate_type = "proc-macro"]

#[cfg(feature = "alloc")]
extern crate alloc;
