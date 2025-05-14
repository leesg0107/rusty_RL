# Rusty RL

A drone reinforcement learning project implemented in Rust. This project uses the `tch` crate (PyTorch bindings for Rust) to implement the PPO (Proximal Policy Optimization) algorithm for drone control.

## Project Structure

- `src/main.rs`: Bevy game engine-based drone simulation
- `src/rl_patch.rs`: Reinforcement learning algorithm (PPO) implementation
- `src/agent.rs`: Drone agent implementation
- `src/environment.rs`: Environment implementation
- `src/rl_plugin.rs`: Integration between RL and Bevy ECS

## Controls

- `TAB`: Switch between control modes (Keyboard / Reinforcement Learning / Circle Test)
- `R`: Reset the drone and target to initial positions
- `T`: Toggle training (in RL mode)
- `S`: Save the current model (in RL mode)

Keyboard control mode:
- `Space`: Ascend
- `Left Shift`: Descend
- Arrow keys: Move horizontally (Forward/Backward/Left/Right)
- `A`/`D`: Rotate left/right

## Current Status

**Note: The reinforcement learning implementation is still a work in progress and may not perform optimally.**

The drone can be controlled manually using keyboard input or by the reinforcement learning agent. The RL algorithm is implemented but still requires tuning and optimization for better performance.

## tch-rs Usage Notes

### Creating Tensors

With tch 0.13.0, use `Tensor::f_from_slice` instead of `Tensor::of_slice`:

```rust
// Error
let tensor = Tensor::of_slice(&[1.0f32, 2.0, 3.0]); 

// Correct
let tensor = Tensor::f_from_slice(&[1.0f32, 2.0, 3.0])?;
```

### Using the pow method

The `pow` method requires a tensor as argument, not a constant:

```rust
// Error
tensor.pow(2.0);

// Correct
let exponent = Tensor::f_from_slice(&[2.0f32])?;
tensor.pow(&exponent);
```

### Using sum_dim_intlist method

Use slices instead of array references with `sum_dim_intlist`:

```rust
// Error
tensor.sum_dim_intlist(&[-1], false, Kind::Float);

// Correct
let dim: &[i64] = &[-1];
tensor.sum_dim_intlist(dim, false, Kind::Float);
```

### Explicit typing for float constants

Float constants should have explicit types:

```rust
// Error
10.0.min(history.len() as f32)

// Correct
10.0f32.min(history.len() as f32)
```

## Bevy Integration

Integrating with Bevy poses thread-safety challenges due to tch tensors not implementing `Send + Sync`. This project addresses this by:

1. Running the learning loop in a separate thread
2. Using thread-safe data structures for communication
3. Using smart pointers and synchronization mechanisms

See `LIBTORCH_BEVY_INTEGRATION.md` for more details.

## Building and Running

```bash
# Main drone simulation
cargo run
```

## PyTorch Setup

Use the `download-libtorch` feature instead of manually installing libtorch:

```toml
# Cargo.toml
[dependencies]
tch = { version = "0.13.0", features = ["download-libtorch"] }
``` 
