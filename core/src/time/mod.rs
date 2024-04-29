/*
   Appellation: epochs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::utils::*;

pub mod epoch;

pub(crate) mod utils {
    pub fn now() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let period = core::time::Duration::from_secs(1);
        let ts = now();
        assert!(ts > 0);
        std::thread::sleep(period);
        assert!(now() > ts);
    }
}
