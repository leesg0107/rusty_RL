cargo clean
cargo update
cargo build
mkdir -p assets
cp quadrotor.urdf assets/
cargo run 