use bevy::prelude::*;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use rand::Rng;
use rand::distributions::{Distribution, Normal};
use std::collections::VecDeque;
use crate::agent::DroneAgent;
use crate::environment::{Environment, Target};
use crate::agent::spawn_keyboard_controlled_drone;

// PPO 하이퍼파라미터
#[derive(Resource)]
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
#[derive(Resource)]
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
        
        // 랜덤 샘플링
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.buffer.len())
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size as usize)
            .cloned()
            .collect();
        
        for &idx in &indices {
            let exp = &self.buffer[idx];
            states.extend_from_slice(&exp.state);
            actions.extend_from_slice(&exp.action);
            rewards.push(exp.reward);
            next_states.extend_from_slice(&exp.next_state);
            dones.push(if exp.done { 1.0 } else { 0.0 });
            log_probs.push(exp.log_prob);
        }
        
        let states = Tensor::of_slice(&states)
            .view([-1, self.buffer[0].state.len() as i64])
            .to_device(device);
        let actions = Tensor::of_slice(&actions)
            .view([-1, self.buffer[0].action.len() as i64])
            .to_device(device);
        let rewards = Tensor::of_slice(&rewards)
            .to_device(device);
        let next_states = Tensor::of_slice(&next_states)
            .view([-1, self.buffer[0].next_state.len() as i64])
            .to_device(device);
        let dones = Tensor::of_slice(&dones)
            .to_device(device);
        let log_probs = Tensor::of_slice(&log_probs)
            .to_device(device);
        
        Some((states, actions, rewards, next_states, dones, log_probs))
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
        let normal = tch::Tensor::randn_like(&mean) * &std + &mean;
        
        // 로그 확률 계산
        let log_prob = -((normal.clone() - mean).pow(2.0) / (2.0 * std.pow(2.0)) + std.log() + std::f64::consts::LN_2PI.sqrt().ln() / 2.0).sum_dim_intlist(&[-1], false, Kind::Float);
        
        (normal, log_prob)
    }
    
    pub fn evaluate_action(&self, x: &Tensor, action: &Tensor) -> (Tensor, Tensor) {
        let (mean, std) = self.forward(x);
        
        // 로그 확률 계산
        let log_prob = -((action - &mean).pow(2.0) / (2.0 * std.pow(2.0)) + std.log() + std::f64::consts::LN_2PI.sqrt().ln() / 2.0).sum_dim_intlist(&[-1], false, Kind::Float);
        
        // 엔트로피 계산
        let entropy = (0.5 + 0.5 * std::f64::consts::LN_2PI + std.log()).sum_dim_intlist(&[-1], false, Kind::Float);
        
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
#[derive(Resource)]
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
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.save(path)?;
        println!("모델 저장 완료: {}", path);
        Ok(())
    }
    
    // 모델 로드
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.vs.load(path)?;
        println!("모델 로드 완료: {}", path);
        Ok(())
    }
}

// 에이전트 메모리 컴포넌트
#[derive(Component, Default)]
pub struct AgentMemory {
    pub prev_state: Option<Vec<f32>>,
    pub prev_action: Option<Vec<f32>>,
    pub prev_log_prob: Option<f32>,
    pub prev_value: Option<f32>,
    pub cumulative_reward: f32,
    pub episode_steps: usize,
    pub total_steps: usize,
    pub total_episodes: usize,
    pub rewards_history: Vec<f32>,
    pub best_reward: f32,
}

// 에이전트 초기화 시스템
pub fn initialize_agent_system(
    mut commands: Commands,
    query: Query<Entity, With<DroneAgent>>,
) {
    for entity in query.iter() {
        commands.entity(entity).insert(AgentMemory::default());
    }
}

// 상태 추출 함수
fn extract_state(agent: &DroneAgent, transform: &Transform, velocity: &Velocity, target_pos: Vec3) -> Vec<f32> {
    let pos = transform.translation;
    let vel = velocity.linvel;
    let target_dir = target_pos - pos;
    
    vec![
        pos.x, pos.y, pos.z,           // 드론 위치
        vel.x, vel.y, vel.z,           // 드론 속도
        target_dir.x, target_dir.y, target_dir.z, // 목표까지 방향
    ]
}

// 행동을 드론 제어 입력으로 변환
fn action_to_control(action: &[f32]) -> (f32, Vec3) {
    let thrust = action[0].clamp(-1.0, 1.0) * 0.5 + 0.5; // 0.0 ~ 1.0 범위로 변환
    let torque = Vec3::new(
        action[1].clamp(-1.0, 1.0),
        action[2].clamp(-1.0, 1.0),
        action[3].clamp(-1.0, 1.0),
    );
    
    (thrust, torque)
}

// 강화학습 시스템
pub fn reinforcement_learning_system(
    mut query: Query<(&mut DroneAgent, &Transform, &Velocity, &mut AgentMemory)>,
    target_query: Query<&Transform, With<Target>>,
    mut buffer: ResMut<ReplayBuffer>,
    network: Res<PPONetwork>,
    params: Res<PPOParams>,
    env: Res<Environment>,
) {
    // 목표 위치 가져오기
    let target_pos = if let Ok(target_transform) = target_query.get_single() {
        target_transform.translation
    } else {
        return; // 목표가 없으면 종료
    };
    
    for (mut agent, transform, velocity, mut memory) in query.iter_mut() {
        // 현재 상태 추출
        let current_state = extract_state(&agent, transform, velocity, target_pos);
        
        // 스텝 카운터 증가
        memory.episode_steps += 1;
        memory.total_steps += 1;
        
        // 이전 상태가 있으면 경험 저장
        if let (Some(prev_state), Some(prev_action), Some(prev_log_prob), Some(prev_value)) = 
            (&memory.prev_state, &memory.prev_action, memory.prev_log_prob, memory.prev_value) {
            
            // 보상 누적
            memory.cumulative_reward += agent.reward;
            
            // 경험 생성
            let experience = Experience {
                state: prev_state.clone(),
                action: prev_action.clone(),
                reward: agent.reward,
                next_state: current_state.clone(),
                done: agent.done,
                log_prob: prev_log_prob,
                value: prev_value,
            };
            
            // 버퍼에 추가
            buffer.add(experience);
        }
        
        // 에피소드 종료 처리
        if agent.done {
            // 에피소드 통계 업데이트
            memory.total_episodes += 1;
            memory.rewards_history.push(memory.cumulative_reward);
            
            if memory.cumulative_reward > memory.best_reward {
                memory.best_reward = memory.cumulative_reward;
            }
            
            // 에피소드 정보 출력 (10 에피소드마다)
            if memory.total_episodes % 10 == 0 {
                println!("=== 학습 통계 ===");
                println!("에피소드: {}", memory.total_episodes);
                println!("총 스텝: {}", memory.total_steps);
                println!("현재 보상: {:.2}", memory.cumulative_reward);
                println!("최고 보상: {:.2}", memory.best_reward);
                
                // 최근 10개 에피소드 평균 보상
                let recent_rewards = memory.rewards_history.iter()
                    .rev().take(10).sum::<f32>() / 10.0.min(memory.rewards_history.len() as f32);
                println!("최근 10개 평균 보상: {:.2}", recent_rewards);
                println!("버퍼 크기: {}", buffer.len());
                println!("학습 단계: {}", network.learning_steps);
                println!("==================");
            }
            
            // 메모리 초기화
            memory.prev_state = None;
            memory.prev_action = None;
            memory.prev_log_prob = None;
            memory.prev_value = None;
            memory.cumulative_reward = 0.0;
            memory.episode_steps = 0;
            
            return;
        }
        
        // 텐서 변환
        let state_tensor = Tensor::of_slice(&current_state)
            .view([1, -1])
            .to_device(params.device);
        
        // 정책 네트워크로 행동 선택
        let (action_tensor, log_prob) = network.actor.sample_action(&state_tensor);
        
        // 가치 네트워크로 상태 가치 예측
        let value = network.critic.forward(&state_tensor).double_value(&[0, 0]) as f32;
        
        // 텐서를 Vec<f32>로 변환
        let action: Vec<f32> = action_tensor
            .view(-1)
            .to_device(Device::Cpu)
            .try_into()
            .unwrap();
        
        // 행동을 드론 제어 입력으로 변환
        let (thrust, torque) = action_to_control(&action);
        
        // 드론에 제어 입력 적용
        agent.thrust = thrust;
        agent.torque = torque;
        
        // 현재 상태와 행동 저장
        memory.prev_state = Some(current_state);
        memory.prev_action = Some(action);
        memory.prev_log_prob = Some(log_prob.double_value(&[0]) as f32);
        memory.prev_value = Some(value);
    }
}

// PPO 업데이트 시스템
pub fn ppo_update_system(
    mut network: ResMut<PPONetwork>,
    mut buffer: ResMut<ReplayBuffer>,
    params: Res<PPOParams>,
    time: Res<Time>,
) {
    // 버퍼에 충분한 데이터가 쌓이면 학습 진행
    if buffer.len() < params.batch_size as usize {
        return;
    }
    
    // 5초에 한 번씩만 업데이트
    if (time.elapsed_seconds() % 5.0) > 0.1 {
        return;
    }
    
    println!("PPO 업데이트 시작 - 버퍼 크기: {}", buffer.len());
    
    // 배치 데이터 가져오기
    if let Some((states, actions, rewards, next_states, dones, old_log_probs)) = 
        buffer.get_batch(params.batch_size, params.device) {
        
        // 여러 에포크 동안 학습
        for _ in 0..params.epochs {
            // 가치 예측
            let values = network.critic.forward(&states);
            let next_values = network.critic.forward(&next_states);
            
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
            let (new_log_probs, entropy) = network.actor.evaluate_action(&states, &actions);
            
            // 비율 계산
            let ratio = (new_log_probs - old_log_probs).exp();
            
            // 클리핑된 서로게이트 목적 함수
            let surrogate1 = &ratio * &advantages;
            let surrogate2 = ratio.clamp(1.0 - params.epsilon, 1.0 + params.epsilon) * &advantages;
            let policy_loss = surrogate1.min_other(&surrogate2).mean(Kind::Float);
            
            // 가치 손실
            let value_pred = network.critic.forward(&states).view(-1);
            let value_loss = (&value_pred - &returns).pow(2.0).mean(Kind::Float);
            
            // 엔트로피 보너스
            let entropy_loss = entropy.mean(Kind::Float);
            
            // 총 손실
            let loss = -policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy_loss;
            
            // 역전파 및 최적화
            network.optimizer.zero_grad();
            loss.backward();
            network.optimizer.step();
        }
        
        // 학습 단계 증가
        network.learning_steps += 1;
        
        // 버퍼 비우기
        buffer.clear();
        
        println!("PPO 업데이트 완료 - 학습 단계: {}", network.learning_steps);
    }
}

// GAE(Generalized Advantage Estimation) 계산
fn compute_gae(rewards: &Tensor, values: &Tensor, next_values: &Tensor, dones: &Tensor, gamma: f64, lambda: f64) -> Tensor {
    let deltas = rewards + gamma * next_values * (1.0 - dones) - values;
    
    let mut advantages = Tensor::zeros_like(deltas);
    let mut gae = Tensor::zeros(&[1], (Kind::Float, deltas.device()));
    
    // 역순으로 GAE 계산
    for i in (0..deltas.size()[0]).rev() {
        let delta_i = deltas.get(i);
        let done_i = dones.get(i);
        
        gae = delta_i + gamma * lambda * (1.0 - done_i) * gae;
        advantages.get(i).copy_(&gae);
    }
    
    advantages
}

// 리셋 확인 시스템
pub fn check_reset(
    mut commands: Commands,
    query: Query<(Entity, &DroneAgent)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target_query: Query<Entity, With<Target>>,
    mut env: ResMut<Environment>,
) {
    for (entity, agent) in query.iter() {
        if agent.done {
            // 이전 드론 제거
            commands.entity(entity).despawn_recursive();
            
            // 이전 타겟 제거
            for target_entity in target_query.iter() {
                commands.entity(target_entity).despawn_recursive();
            }
            
            // 새 드론 생성
            let drone_entity = spawn_keyboard_controlled_drone(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, 2.0, 0.0));
            
            // 메모리 컴포넌트 추가
            commands.entity(drone_entity).insert(AgentMemory::default());
            
            // 새 목표 생성 (랜덤 위치)
            let mut rng = rand::thread_rng();
            let target_pos = Vec3::new(
                rng.gen_range(-5.0..5.0),
                rng.gen_range(3.0..8.0),
                rng.gen_range(-5.0..5.0),
            );
            
            spawn_target(&mut commands, &mut meshes, &mut materials, target_pos);
            
            // 환경 업데이트
            env.episode += 1;
            env.step = 0;
            env.target_position = target_pos;
            
            println!("에피소드 리셋: {} - 새 목표 위치: {:?}", env.episode, target_pos);
            break;
        }
    }
}

// 모델 저장 시스템
pub fn save_model_system(
    keyboard_input: Res<Input<KeyCode>>,
    network: Res<PPONetwork>,
) {
    // S 키를 눌러 모델 저장
    if keyboard_input.just_pressed(KeyCode::S) {
        let path = "models/drone_ppo_latest.pt";
        if let Err(e) = network.save(path) {
            println!("모델 저장 실패: {}", e);
        }
    }
}

// 모델 로드 시스템
pub fn load_model_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut network: ResMut<PPONetwork>,
) {
    // L 키를 눌러 모델 로드
    if keyboard_input.just_pressed(KeyCode::L) {
        let path = "models/drone_ppo_latest.pt";
        if let Err(e) = network.load(path) {
            println!("모델 로드 실패: {}", e);
        }
    }
} 