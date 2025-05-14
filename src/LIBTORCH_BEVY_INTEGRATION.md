# Integrating libtorch with Bevy

This document explains the challenges and solutions encountered when integrating libtorch (PyTorch C++ library) with Bevy, a Rust-based game engine.

## Key Challenges

The main challenges in integrating libtorch with Bevy are:

1. **Thread-safety issues**: Types like `Tensor` and `VarStore` in libtorch contain raw pointers such as `*mut C_tensor` that are not thread-safe and therefore don't implement the `Send + Sync` traits. However, Bevy's resource system requires these traits.

2. **API version differences**: The API can change significantly between different versions of the tch-rs crate.

3. **Memory management**: PyTorch tensors manage memory in C++, which may not align well with Rust's ownership model.

## Solutions

### 1. Running the Learning Loop in a Separate Thread

Instead of performing tensor operations directly in Bevy's ECS systems, run the learning loop in a separate thread:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// Data structure that can be shared between threads
struct SharedData {
    experiences: Vec<Experience>,
    model_weights: Vec<f32>,
}

// Thread-safe wrapper for use as a Bevy resource
#[derive(Resource)]
struct ModelResource {
    shared_data: Arc<Mutex<SharedData>>,
    training_thread: Option<thread::JoinHandle<()>>,
}

// Create a training thread when the system starts
fn setup_training_thread(mut commands: Commands) {
    let shared_data = Arc::new(Mutex::new(SharedData {
        experiences: Vec::new(),
        model_weights: Vec::new(),
    }));
    
    let thread_data = shared_data.clone();
    let training_thread = thread::spawn(move || {
        // Free to use libtorch in this thread
        let mut network = PPONetwork::default();
        let mut buffer = ReplayBuffer::default();
        
        loop {
            // Get data
            let experiences = {
                let mut data = thread_data.lock().unwrap();
                std::mem::take(&mut data.experiences)
            };
            
            // Perform training...
            
            // Store results
            {
                let mut data = thread_data.lock().unwrap();
                // Update model weights...
            }
            
            thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    
    commands.insert_resource(ModelResource {
        shared_data,
        training_thread: Some(training_thread),
    });
}

// System to add experience data
fn collect_experience_system(
    model: Res<ModelResource>,
    // Other queries...
) {
    // Collect experience from agents...
    
    // Add experience to shared data
    if let Ok(mut data) = model.shared_data.lock() {
        data.experiences.push(experience);
    }
}

// System to perform inference using model weights
fn inference_system(
    model: Res<ModelResource>,
    // Other queries...
) {
    // Get model weights from shared data
    let weights = if let Ok(data) = model.shared_data.lock() {
        data.model_weights.clone()
    } else {
        return;
    };
    
    // Perform simple inference using weights (without libtorch)...
}
```

### 2. Using a Single-Threaded Scheduler

Use Bevy's single-threaded scheduler instead of the multi-threaded one:

```rust
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            // Use single-threaded scheduler
            bevy::core::TaskPoolPlugin::SingleThreaded, 
        ))
        // ... other configuration
        .run();
}
```

### 3. Implementing Safe Wrappers

Use safe types like `Vec<f32>` as Bevy resources instead of Tensor, and only convert to Tensor when actual learning is needed:

```rust
#[derive(Resource)]
struct SafeModelWeights {
    layers: Vec<Vec<f32>>,
    shapes: Vec<Vec<i64>>,
}

fn learning_system(
    weights: Res<SafeModelWeights>,
    // ... other parameters
) {
    // Safely work within Bevy systems
    
    // Only use tensors in an isolated scope when needed
    {
        // Convert safe data to tensors
        let mut tensors = Vec::new();
        for (layer, shape) in weights.layers.iter().zip(weights.shapes.iter()) {
            let tensor = Tensor::f_from_slice(&layer).unwrap()
                .view(shape.as_slice());
            tensors.push(tensor);
        }
        
        // Perform tensor operations...
        
        // Convert results back to safe form
    }
}
```

## Recommended Patterns

1. **Minimize Bevy-libtorch interaction**: Minimize direct interaction between Bevy and libtorch, using an intermediate layer for data transfer.

2. **Separate learning and inference**: Separate the learning part into a dedicated thread and only perform inference in Bevy systems.

3. **Use a stable version**: Use the most stable version of tch-rs that provides the features needed for your project (e.g., 0.7.2).

4. **Use the download-libtorch feature**: Instead of manually installing libtorch, use the `download-libtorch` feature of tch-rs:

```toml
[dependencies]
tch = { version = "0.13.0", features = ["download-libtorch"] }
```

## Example Project Structure

```
rusty_RL/
├── src/
│   ├── main.rs        # Bevy app setup and execution
│   ├── agent.rs       # Agent implementation (Bevy ECS components)
│   ├── environment.rs # Environment implementation (Bevy ECS systems)
│   ├── rl/            # Reinforcement learning module (using libtorch)
│   │   ├── mod.rs     # Module declarations and safe interface
│   │   ├── model.rs   # Model implementation (PPO, etc.)
│   │   └── buffer.rs  # Experience buffer implementation
│   └── interface.rs   # Safe interface between Bevy and RL modules
├── Cargo.toml         # Dependencies
└── README.md
```

This structure clearly separates code using libtorch from Bevy ECS code, integrating them safely only when necessary. 