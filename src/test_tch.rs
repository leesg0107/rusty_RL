use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tch 0.13.0 functionality");
    
    // Create a simple tensor
    let tensor = Tensor::f_from_slice(&[1.0f32, 2.0, 3.0])?;
    println!("Tensor: {:?}", tensor);
    
    // Create a simple neural network
    let vs = nn::VarStore::new(Device::Cpu);
    let net = nn::seq()
        .add(nn::linear(&vs.root(), 3, 1, Default::default()));
    
    println!("Network created successfully");
    
    // Test tensor operations
    let tensor2 = Tensor::f_from_slice(&[4.0f32, 5.0, 6.0])?;
    let sum = &tensor + &tensor2;
    println!("Sum: {:?}", sum);
    
    // Test creating optimizer
    let optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    println!("Optimizer created successfully");
    
    println!("All tests passed!");
    Ok(())
} 