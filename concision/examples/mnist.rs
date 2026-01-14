/*
    Appellation: mnist <module>
    Created At: 2026.01.13:22:06:22
    Contrib: @FL03
*/
extern crate concision as cnc;

fn main() -> cnc::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    // load the dataset
    // let dataset =
    Ok(())
}
