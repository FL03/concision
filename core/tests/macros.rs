/*
    Appellation: macros <module>
    Created At: 2025.12.13:07:32:40
    Contrib: @FL03
*/
#![cfg(feature = "macros")]

use concision_core::config;

config! {
    TestConfig {
        test_key: f32,
        number_key: usize,
    }
}

#[test]
fn test_config_macro() {
    let cfg = TestConfig::new().with_test_key(3.14);
    assert_eq!(cfg.test_key(), &3.14);
}
