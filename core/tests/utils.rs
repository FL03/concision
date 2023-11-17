#[cfg(test)]
extern crate concision_core;

use concision_core::prelude::now;

#[test]
fn test_timestamp() {
    let period = std::time::Duration::from_secs(1);
    let ts = now();
    assert!(ts > 0);
    std::thread::sleep(period);
    assert!(now() > ts);
}
