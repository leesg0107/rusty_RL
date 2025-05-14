use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use std::collections::VecDeque;
use std::error::Error;
use std::f64::consts::PI;

// PPO 하이퍼파라미터
pub struct PPOParams {
    pub learning_rate: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub epsilon: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub max_grad_norm: f64,
    pub batch_size: i64,
    pub epochs: i64,
    pub state_dim: i64,
    pub action_dim: i64,
    pub hidden_dim: i64,
    pub device: Device,
}

impl Default for PPOParams {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            lambda: 0.95,
            epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 0.5,
            batch_size: 64,
            epochs: 10,
            state_dim: 9,  // 드론 위치(3), 속도(3), 목표까지 거리(3)
            action_dim: 4, // 추력(1), 토크(3)
            hidden_dim: 64,
            device: Device::Cpu, // GPU가 있으면 Device::Cuda(0)으로 변경
        }
    }
}

// 경험 저장을 위한 구조체
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub log_prob: f32,
    pub value: f32,
}

// 리플레이 버퍼
pub struct ReplayBuffer {
    pub buffer: VecDeque<Experience>,
    pub capacity: usize,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity: 10000,
        }
    }
}

impl ReplayBuffer {
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }
    
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    // 배치 데이터 추출
    pub fn get_batch(&self, batch_size: i64, device: Device) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.buffer.len() < batch_size as usize {
            return None;
        }
        
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut dones = Vec::new();
        let mut log_probs = Vec::new();
        
        // 랜덤 샘플링 (여기서는 간소화를 위해 처음 batch_size개만 사용)
        let indices: Vec<usize> = (0..batch_size as usize).collect();
        
        for &idx in &indices {
            let exp = &self.buffer[idx];
            states.extend_from_slice(&exp.state);
            actions.extend_from_slice(&exp.action);
            rewards.push(exp.reward);
            next_states.extend_from_slice(&exp.next_state);
            dones.push(if exp.done { 1.0 } else { 0.0 });
            log_probs.push(exp.log_prob);
        }
        
        // 텐서 생성 시 f_from_slice 사용 (0.13.0 API)
        let states_result = Tensor::f_from_slice(&states)
            .map(|t| t.view([-1, self.buffer[0].state.len() as i64]).to_device(device));
            
        let actions_result = Tensor::f_from_slice(&actions)
            .map(|t| t.view([-1, self.buffer[0].action.len() as i64]).to_device(device));
            
        let rewards_result = Tensor::f_from_slice(&rewards)
            .map(|t| t.to_device(device));
            
        let next_states_result = Tensor::f_from_slice(&next_states)
            .map(|t| t.view([-1, self.buffer[0].next_state.len() as i64]).to_device(device));
            
        let dones_result = Tensor::f_from_slice(&dones)
            .map(|t| t.to_device(device));
            
        let log_probs_result = Tensor::f_from_slice(&log_probs)
            .map(|t| t.to_device(device));
        
        // 모든 변환이 성공했는지 확인
        if let (Ok(states), Ok(actions), Ok(rewards), Ok(next_states), Ok(dones), Ok(log_probs)) = 
            (states_result, actions_result, rewards_result, next_states_result, dones_result, log_probs_result) {
            Some((states, actions, rewards, next_states, dones, log_probs))
        } else {
            None
        }
    }
}

// 액터 네트워크
pub struct Actor {
    pub network: nn::Sequential,
    pub mean_layer: nn::Linear,
    pub log_std: Tensor,
}

impl Actor {
    pub fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let network = nn::seq()
            .add(nn::linear(vs / "actor_l1", state_dim, hidden_dim, Default::default()))
            .add_fn(|xs| xs.tanh());
        
        let mean_layer = nn::linear(vs / "actor_mean", hidden_dim, action_dim, Default::default());
        let log_std = vs.zeros("actor_log_std", &[action_dim]);
        
        Self {
            network,
            mean_layer,
            log_std,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let hidden = self.network.forward(x);
        let mean = self.mean_layer.forward(&hidden);
        let std = self.log_std.exp();
        
        (mean, std)
    }
    
    pub fn sample_action(&self, x: &Tensor) -> (Tensor, Tensor) {
        let (mean, std) = self.forward(x);
        
        // 정규 분포에서 샘플링
        let noise = Tensor::randn_like(&mean);
        let action = &mean + &std * &noise;
        
        // 로그 확률 계산 - Tensor 연산 사용
        let var = &std * &std;
        let log_scale = &std.log();
        let dim: &[i64] = &[-1]; // Use slice instead of array
        let log_prob = -(&noise.pow(&Tensor::f_from_slice(&[2.0f32]).unwrap()) / 2.0 
                        + log_scale
                        + Tensor::f_from_slice(&[2.0 * std::f32::consts::PI]).unwrap().log() / 2.0)
                        .sum_dim_intlist(dim, false, Kind::Float);
        
        (action, log_prob)
    }
    
    pub fn evaluate_action(&self, x: &Tensor, action: &Tensor) -> (Tensor, Tensor) {
        let (mean, std) = self.forward(x);
        
        // 로그 확률 계산 - Tensor 연산 사용
        let var = &std * &std;
        let log_scale = &std.log();
        let scaled_diff = (action - &mean) / &std;
        let dim: &[i64] = &[-1]; // Use slice instead of array
        let log_prob = -(&scaled_diff.pow(&Tensor::f_from_slice(&[2.0f32]).unwrap()) / 2.0 
                        + log_scale
                        + Tensor::f_from_slice(&[2.0 * std::f32::consts::PI]).unwrap().log() / 2.0)
                        .sum_dim_intlist(dim, false, Kind::Float);
        
        // 엔트로피 계산 - (PI*2.0).ln() 사용
        let entropy = (0.5 + 0.5 * (PI*2.0).ln() as f32 + &std.log())
                        .sum_dim_intlist(dim, false, Kind::Float);
        
        (log_prob, entropy)
    }
}

// 크리틱 네트워크
pub struct Critic {
    pub network: nn::Sequential,
}

impl Critic {
    pub fn new(vs: &nn::Path, state_dim: i64, hidden_dim: i64) -> Self {
        let network = nn::seq()
            .add(nn::linear(vs / "critic_l1", state_dim, hidden_dim, Default::default()))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs / "critic_l2", hidden_dim, 1, Default::default()));
        
        Self {
            network,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.network.forward(x)
    }
}

// PPO 네트워크
pub struct PPONetwork {
    pub vs: nn::VarStore,
    pub actor: Actor,
    pub critic: Critic,
    pub optimizer: nn::Optimizer,
    pub learning_steps: usize,
}

impl Default for PPONetwork {
    fn default() -> Self {
        let params = PPOParams::default();
        let vs = nn::VarStore::new(params.device);
        let root = vs.root();
        
        let actor = Actor::new(&root, params.state_dim, params.action_dim, params.hidden_dim);
        let critic = Critic::new(&root, params.state_dim, params.hidden_dim);
        
        let optimizer = nn::Adam::default()
            .build(&vs, params.learning_rate)
            .unwrap();
        
        Self {
            vs,
            actor,
            critic,
            optimizer,
            learning_steps: 0,
        }
    }
}

impl PPONetwork {
    // 모델 저장
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        println!("모델 저장 완료: {}", path);
        Ok(())
    }
    
    // 모델 로드
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.vs.load(path)?;
        println!("모델 로드 완료: {}", path);
        Ok(())
    }
    
    // PPO 업데이트
    pub fn update(&mut self, buffer: &mut ReplayBuffer, params: &PPOParams) -> bool {
        // 버퍼에 충분한 데이터가 쌓이면 학습 진행
        if buffer.len() < params.batch_size as usize {
            return false;
        }
        
        println!("PPO 업데이트 시작 - 버퍼 크기: {}", buffer.len());
        
        // 배치 데이터 가져오기
        if let Some((states, actions, rewards, next_states, dones, old_log_probs)) = 
            buffer.get_batch(params.batch_size, params.device) {
            
            // 여러 에포크 동안 학습
            for _ in 0..params.epochs {
                // 가치 예측
                let values = self.critic.forward(&states);
                let next_values = self.critic.forward(&next_states);
                
                // 어드밴티지 계산 (GAE)
                let advantages = compute_gae(
                    &rewards, 
                    &values.view(-1), 
                    &next_values.view(-1), 
                    &dones, 
                    params.gamma, 
                    params.lambda
                );
                
                // 리턴 계산
                let returns = &advantages + &values.view(-1);
                
                // 정책 업데이트
                let (new_log_probs, entropy) = self.actor.evaluate_action(&states, &actions);
                
                // 비율 계산
                let ratio = (&new_log_probs - &old_log_probs).exp();
                
                // 클리핑된 서로게이트 목적 함수
                let surrogate1 = &ratio * &advantages;
                let surrogate2 = ratio.clamp(1.0 - params.epsilon, 1.0 + params.epsilon) * &advantages;
                let policy_loss = surrogate1.min_other(&surrogate2).mean(Kind::Float);
                
                // 가치 손실 
                let value_pred = self.critic.forward(&states).view(-1);
                let two_tensor = Tensor::f_from_slice(&[2.0f32]).unwrap();
                let value_loss = (&value_pred - &returns).pow(&two_tensor).mean(Kind::Float);
                
                // 엔트로피 보너스
                let entropy_loss = entropy.mean(Kind::Float);
                
                // 총 손실
                let loss = -policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy_loss;
                
                // 역전파 및 최적화
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            }
            
            // 학습 단계 증가
            self.learning_steps += 1;
            
            // 버퍼 비우기
            buffer.clear();
            
            println!("PPO 업데이트 완료 - 학습 단계: {}", self.learning_steps);
            return true;
        }
        
        false
    }
}

// GAE(Generalized Advantage Estimation) 계산 - 더 간단한 구현
fn compute_gae(rewards: &Tensor, values: &Tensor, next_values: &Tensor, dones: &Tensor, gamma: f64, lambda: f64) -> Tensor {
    // 원래 디바이스 저장
    let device = rewards.device();
    
    // 스칼라 값을 텐서로 변환
    let gamma_t = gamma as f32;
    let lambda_t = lambda as f32;
    
    // 데이터를 벡터로 변환하기 전에 CPU로 이동
    let rewards_cpu = rewards.to_device(Device::Cpu);
    let values_cpu = values.to_device(Device::Cpu);
    let next_values_cpu = next_values.to_device(Device::Cpu);
    let dones_cpu = dones.to_device(Device::Cpu);
    
    // f32 데이터 생성 
    let mut advantages = vec![0.0f32; rewards_cpu.size()[0] as usize];
    
    // 현재 버전에서는 텐서를 직접 벡터로 가져올 수 없으므로 각 요소를 루프로 읽어옴
    let batch_size = rewards_cpu.size()[0] as usize;
    let mut rewards_vec = vec![0.0f32; batch_size];
    let mut values_vec = vec![0.0f32; batch_size];
    let mut next_values_vec = vec![0.0f32; batch_size]; 
    let mut dones_vec = vec![0.0f32; batch_size];
    
    for i in 0..batch_size {
        rewards_vec[i] = rewards_cpu.double_value(&[i as i64]) as f32;
        values_vec[i] = values_cpu.double_value(&[i as i64]) as f32;
        next_values_vec[i] = next_values_cpu.double_value(&[i as i64]) as f32;
        dones_vec[i] = dones_cpu.double_value(&[i as i64]) as f32;
    }
    
    // GAE 계산
    let mut last_gae = 0.0;
    for i in (0..batch_size).rev() {
        let delta = rewards_vec[i] + gamma_t * next_values_vec[i] * (1.0 - dones_vec[i]) - values_vec[i];
        last_gae = delta + gamma_t * lambda_t * (1.0 - dones_vec[i]) * last_gae;
        advantages[i] = last_gae;
    }
    
    // 계산된 어드밴티지를 다시 텐서로 변환
    let advantages_tensor = Tensor::f_from_slice(&advantages).unwrap();
    
    // 원래 디바이스로 되돌리기
    advantages_tensor.to_device(device)
}

// 테스트 함수
pub fn test_ppo() -> Result<(), Box<dyn Error>> {
    println!("Testing PPO implementation");
    
    let params = PPOParams::default();
    let mut network = PPONetwork::default();
    let mut buffer = ReplayBuffer::default();
    
    // 간단한 경험 추가
    for i in 0..100 {
        let state = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let action = vec![0.1, 0.2, 0.3, 0.4];
        let reward = 0.5;
        let next_state = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let done = i % 20 == 19; // 매 20번째마다 에피소드 종료
        
        // 상태를 텐서로 변환
        let state_tensor = Tensor::f_from_slice(&state)?.view([1, -1]);
        
        // 액션 샘플링
        let (_action_tensor, log_prob) = network.actor.sample_action(&state_tensor);
        
        // 가치 예측
        let value = network.critic.forward(&state_tensor).double_value(&[0, 0]) as f32;
        
        // 경험 생성
        let experience = Experience {
            state,
            action: action.clone(),
            reward,
            next_state,
            done,
            log_prob: log_prob.double_value(&[0]) as f32,
            value,
        };
        
        // 버퍼에 추가
        buffer.add(experience);
    }
    
    println!("Buffer size: {}", buffer.len());
    
    // PPO 업데이트 수행
    let updated = network.update(&mut buffer, &params);
    println!("Update performed: {}", updated);
    
    // 모델 저장 및 로드는 현재 작동하지 않으므로 생략
    /* 
    network.save("test_model.pt")?;
    network.load("test_model.pt")?;
    */
    
    println!("Test completed successfully!");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    test_ppo()
} 