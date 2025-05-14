// src/rl_patch.rs
// This file contains patches for rl.rs that fix the major issues with tch-rs integration with Bevy

use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::VecDeque;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use bevy::prelude::*;
use std::time::Duration;
use rand::prelude::*;

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

// 하이퍼파라미터 구조체 (스레드 간 공유)
#[derive(Resource, Clone)]
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
            learning_rate: 1e-4,
            gamma: 0.999,
            lambda: 0.97,
            epsilon: 0.1,
            value_coef: 0.5,
            entropy_coef: 0.005,
            max_grad_norm: 0.5,
            batch_size: 256,
            epochs: 3,
            state_dim: 9,
            action_dim: 4,
            hidden_dim: 256,
            device: Device::Cpu,
        }
    }
}

// 경험 데이터 구조체 (Bevy와 학습 스레드 간 공유)
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

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity,
        }
    }
    
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
}

// 스레드 간 공유할 안전한 구조체
pub struct SharedRLData {
    pub experiences: Vec<Experience>,
    pub model_parameters: Vec<Vec<f32>>,
    pub action_output: Vec<f32>,
    pub value_output: f32,
    pub log_prob: f32,
    pub state_for_inference: Vec<f32>,
    pub is_training: bool,
    pub training_steps: usize,
    pub best_reward: f32,
    pub recent_rewards: Vec<f32>,
    pub save_requested: bool,
    pub terminate: bool,
}

impl Default for SharedRLData {
    fn default() -> Self {
        Self {
            experiences: Vec::new(),
            model_parameters: Vec::new(),
            action_output: vec![0.0; 4], // 기본 액션(0, 0, 0, 0)
            value_output: 0.0,
            log_prob: 0.0,
            state_for_inference: Vec::new(),
            is_training: false,
            training_steps: 0,
            best_reward: 0.0,
            recent_rewards: Vec::new(),
            save_requested: false,
            terminate: false,
        }
    }
}

// Bevy 리소스로 사용할 안전한 래퍼
#[derive(Resource)]
pub struct RLThread {
    pub shared_data: Arc<Mutex<SharedRLData>>,
    pub training_thread: Option<thread::JoinHandle<()>>,
    pub is_running: bool,
}

impl Default for RLThread {
    fn default() -> Self {
        Self {
            shared_data: Arc::new(Mutex::new(SharedRLData::default())),
            training_thread: None,
            is_running: false,
        }
    }
}

// 학습 스레드 시작 함수
pub fn start_training_thread(rl_thread: &mut RLThread, params: PPOParams) {
    if rl_thread.is_running {
        return; // 이미 실행 중이면 종료
    }
    
    // 공유 데이터 클론
    let shared_data = rl_thread.shared_data.clone();
    
    // 학습 스레드 생성
    let training_thread = thread::spawn(move || {
        println!("RL 학습 스레드 시작");
        
        // 여기서 tch-rs 관련 코드 실행 (Bevy 스레드와 격리됨)
        let vs = nn::VarStore::new(params.device);
        let actor = create_actor(&vs.root(), params.state_dim, params.action_dim, params.hidden_dim);
        let critic = create_critic(&vs.root(), params.state_dim, params.hidden_dim);
        let mut optimizer = nn::Adam::default()
            .build(&vs, params.learning_rate)
            .expect("Failed to create optimizer");
        
        let mut buffer = ReplayBuffer::new(10000);
        let mut learning_steps = 0;
        
        loop {
            // 50ms 대기 (CPU 사용량 감소)
            thread::sleep(Duration::from_millis(50));
            
            // 공유 데이터에서 경험 가져오기
            let mut request_inference = false;
            let state_for_inference: Vec<f32>;
            let should_train: bool;
            let experiences: Vec<Experience>;
            
            {
                let mut data = shared_data.lock().unwrap();
                experiences = std::mem::take(&mut data.experiences);
                should_train = data.is_training && buffer.len() >= params.batch_size as usize;
                
                // 학습 통계 업데이트
                data.training_steps = learning_steps;
                
                // 인퍼런스 요청이 있는지 확인 (상태값이 있으면)
                if !data.state_for_inference.is_empty() {
                    request_inference = true;
                    state_for_inference = data.state_for_inference.clone();
                    data.state_for_inference.clear();
                } else {
                    state_for_inference = Vec::new();
                }
            }
            
            // 수집된 경험을 버퍼에 추가
            for exp in experiences {
                buffer.add(exp);
            }
            
            // 인퍼런스 요청이 있으면 처리
            if request_inference {
                let action_value = perform_inference(&actor, &critic, &state_for_inference, params.device, learning_steps);
                
                // 결과를 공유 데이터에 저장
                if let Ok(mut data) = shared_data.lock() {
                    data.action_output = action_value.0;
                    data.value_output = action_value.1;
                    data.log_prob = action_value.2;
                }
            }
            
            // 학습 조건 충족 시 PPO 업데이트 수행
            if should_train {
                // PPO 업데이트
                if let Some((states, actions, rewards, next_states, dones, old_log_probs)) = 
                    get_batch_fixed(&buffer, params.batch_size, params.device) {
                    
                    // 여러 에포크 동안 학습
                    for _ in 0..params.epochs {
                        // 가치 예측
                        let values = critic.forward(&states);
                        let next_values = critic.forward(&next_states);
                        
                        // 어드밴티지 계산 (GAE)
                        let advantages = compute_gae_fixed(
                            &rewards, 
                            &values.view(-1), 
                            &next_values.view(-1), 
                            &dones, 
                            params.gamma, 
                            params.lambda
                        );
                        
                        // 리턴 계산
                        let returns = &advantages + &values.view(-1);
                        
                        // Advantage 정규화 추가 (평균 0, 표준편차 1)
                        let advantages_mean = advantages.mean(Kind::Float);
                        let two_tensor = Tensor::f_from_slice(&[2.0f32]).unwrap().to_device(advantages.device());
                        let advantages_std = (&advantages - &advantages_mean).pow(&two_tensor).mean(Kind::Float).sqrt() + 1e-8;
                        let normalized_advantages = (&advantages - &advantages_mean) / &advantages_std;
                        
                        // 정책 업데이트
                        let (new_log_probs, entropy) = evaluate_action_fixed(&actor, &states, &actions);
                        
                        // 비율 계산
                        let ratio = (&new_log_probs - &old_log_probs).exp();
                        
                        // 클리핑된 서로게이트 목적 함수
                        let surrogate1 = &ratio * &normalized_advantages;
                        let surrogate2 = ratio.clamp(1.0 - params.epsilon, 1.0 + params.epsilon) * &normalized_advantages;
                        let policy_loss = surrogate1.min_other(&surrogate2).mean(Kind::Float);
                        
                        // 가치 손실
                        let value_pred = critic.forward(&states).view(-1);
                        let value_loss = value_loss_fixed(&value_pred, &returns);
                        
                        // 엔트로피 보너스
                        let entropy_loss = entropy.mean(Kind::Float);
                        
                        // 총 손실
                        let loss = -policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy_loss;
                        
                        // 역전파 및 최적화
                        optimizer.zero_grad();
                        loss.backward();
                        optimizer.step();
                    }
                    
                    // 학습 단계 증가
                    learning_steps += 1;
                    
                    // 버퍼 비우기
                    buffer.clear();
                    
                    println!("PPO 업데이트 완료 - 학습 단계: {}", learning_steps);
                }
            }
            
            // 모델 저장 요청이 있는지 확인
            let should_save = {
                let data = shared_data.lock().unwrap();
                data.save_requested
            };
            
            if should_save {
                let path = "models/drone_ppo_latest.pt";
                if let Err(e) = vs.save(path) {
                    println!("모델 저장 실패: {}", e);
                } else {
                    println!("모델 저장 완료: {}", path);
                    if let Ok(mut data) = shared_data.lock() {
                        data.save_requested = false;
                    }
                }
            }
            
            // 스레드 종료 요청이 있는지 확인
            let should_terminate = {
                let data = shared_data.lock().unwrap();
                data.terminate
            };
            
            if should_terminate {
                println!("RL 학습 스레드 종료");
                break;
            }
        }
    });
    
    rl_thread.training_thread = Some(training_thread);
    rl_thread.is_running = true;
}

// 경험 추가 함수 (Bevy 시스템에서 호출)
pub fn add_experience(rl_thread: &RLThread, experience: Experience) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.experiences.push(experience);
    }
}

// 액션 요청 함수 (Bevy 시스템에서 호출)
pub fn request_action(rl_thread: &RLThread, state: Vec<f32>) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.state_for_inference = state;
    }
}

// 액션 결과 가져오기 (Bevy 시스템에서 호출)
pub fn get_action_result(rl_thread: &RLThread) -> Option<(Vec<f32>, f32, f32)> {
    if let Ok(data) = rl_thread.shared_data.lock() {
        if !data.action_output.is_empty() {
            let is_all_zero = data.action_output.iter().all(|&val| val == 0.0);
            
            // 디버그 로그 추가
            if is_all_zero {
                println!("경고: 모든 액션 값이 0.0입니다. 학습이 올바르게 진행되지 않을 수 있습니다.");
            }
            
            return Some((data.action_output.clone(), data.value_output, data.log_prob));
        } else {
            println!("경고: action_output이 비어 있습니다.");
        }
    } else {
        println!("경고: shared_data 락을 얻을 수 없습니다.");
    }
    None
}

// 학습 시작/중지 (Bevy 시스템에서 호출)
pub fn set_training_enabled(rl_thread: &RLThread, enabled: bool) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.is_training = enabled;
    }
}

// 리워드 정보 업데이트 (Bevy 시스템에서 호출)
pub fn update_reward_info(rl_thread: &RLThread, episode_reward: f32) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.recent_rewards.push(episode_reward);
        if data.recent_rewards.len() > 100 {
            data.recent_rewards.remove(0);
        }
        if episode_reward > data.best_reward {
            data.best_reward = episode_reward;
        }
    }
}

// 모델 저장 요청 (Bevy 시스템에서 호출)
pub fn request_save_model(rl_thread: &RLThread) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.save_requested = true;
    }
}

// 학습 스레드 종료 요청 (Bevy 시스템에서 호출)
pub fn request_terminate(rl_thread: &RLThread) {
    if let Ok(mut data) = rl_thread.shared_data.lock() {
        data.terminate = true;
    }
}

// 학습 상태 정보 가져오기 (Bevy 시스템에서 호출)
pub fn get_training_stats(rl_thread: &RLThread) -> (usize, f32, f32) {
    if let Ok(data) = rl_thread.shared_data.lock() {
        let avg_reward = if data.recent_rewards.is_empty() {
            0.0
        } else {
            data.recent_rewards.iter().sum::<f32>() / data.recent_rewards.len() as f32
        };
        (data.training_steps, data.best_reward, avg_reward)
    } else {
        (0, 0.0, 0.0)
    }
}

// === 이하 tch 관련 유틸리티 함수들 ===

fn create_actor(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "actor_l1", state_dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "actor_l2", hidden_dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "actor_l3", hidden_dim, hidden_dim / 2, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "actor_mean", hidden_dim / 2, action_dim, Default::default()))
}

fn create_critic(vs: &nn::Path, state_dim: i64, hidden_dim: i64) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "critic_l1", state_dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "critic_l2", hidden_dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "critic_l3", hidden_dim, hidden_dim / 2, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "critic_l4", hidden_dim / 2, 1, Default::default()))
}

// 인퍼런스 수행 함수
fn perform_inference(
    actor: &nn::Sequential, 
    critic: &nn::Sequential, 
    state: &[f32], 
    device: Device,
    training_steps: usize
) -> (Vec<f32>, f32, f32) {
    let state_tensor = Tensor::f_from_slice(state).unwrap()
        .view([1, -1])
        .to_device(device);
    
    // 액터 네트워크 실행 (행동 생성)
    let mean = actor.forward(&state_tensor);
    
    // 학습 진행도에 따라 표준편차 점진적 감소 (0.5 -> 0.1)
    let initial_std = 0.5f32;
    let final_std = 0.1f32;
    let decay_steps = 5000.0;
    let std_value = final_std + (initial_std - final_std) * 
                   ((-1.0 * training_steps as f32 / decay_steps).exp());
    
    let std = Tensor::f_from_slice(&[std_value; 4]).unwrap().to_device(device);
    
    // 가치 네트워크 실행 (가치 예측)
    let value = critic.forward(&state_tensor).double_value(&[0, 0]) as f32;
    
    // 랜덤 샘플링
    let noise = Tensor::randn_like(&mean);
    
    // 초기 단계에는 특별한 행동 적용
    let action = if training_steps < 500 {
        // 첫 500 스텝 동안 수정된 행동 적용
        let mean_cpu = mean.to_device(Device::Cpu);
        match mean_cpu.view(-1).double_value(&[0]) {
            v => {
                // 추력은 항상 양수값을 유지하고 기본값은 0.5 이상이 되도록
                let thrust = 0.8f32.max(v as f32);
                
                // 나머지 제어 값들은 작게 유지
                let mut action_values = Vec::new();
                for i in 0..4 {
                    if i == 0 {
                        action_values.push(thrust);
                    } else {
                        // 다른 제어 값들은 작게 유지 (토크 최소화)
                        let control_value = mean_cpu.view(-1).double_value(&[i as i64]) as f32;
                        action_values.push(control_value * 0.2); // 토크 값 감소
                    }
                }
                
                let fixed_mean = Tensor::f_from_slice(&action_values).unwrap().to_device(device);
                &fixed_mean + &std * &noise * 0.3 // 노이즈도 30%로 감소
            }
        }
    } else {
        &mean + &std * &noise
    };
    
    // 로그 확률 계산
    let two_tensor = Tensor::f_from_slice(&[2.0f32]).unwrap().to_device(device);
    let pi_const = Tensor::f_from_slice(&[2.0 * std::f32::consts::PI]).unwrap().log() / 2.0;
    let log_prob = -(&noise.pow(&two_tensor) / 2.0 + &std.log() + pi_const).sum(Kind::Float).double_value(&[]) as f32;
    
    // 텐서를 Vec<f32>로 변환
    let action_vec: Vec<f32> = action
        .view(-1)
        .to_device(Device::Cpu)
        .try_into()
        .unwrap();
    
    // 디버깅: 생성된 액션 확인 (가끔)
    if rand::random::<f32>() < 0.01 { // 1% 확률로 출력
        let mean_vec: Vec<f32> = match mean.to_device(Device::Cpu).try_into() {
            Ok(v) => v,
            Err(_) => vec![0.0; 4], // 변환 실패 시 기본값
        };
        
        println!("상태: {:?}, 평균: {:?}, 액션: {:?}, 가치: {:.3}, std: {:.3}", 
                 &state[..3], // 위치만 출력 (간결함을 위해)
                 mean_vec,
                 &action_vec,
                 value,
                 std_value);
    }
    
    (action_vec, value, log_prob)
}

// 배치 데이터 추출
pub fn get_batch_fixed(buffer: &ReplayBuffer, batch_size: i64, device: Device) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    if buffer.len() < batch_size as usize {
        return None;
    }
    
    let mut states = Vec::new();
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    let mut next_states = Vec::new();
    let mut dones = Vec::new();
    let mut log_probs = Vec::new();
    
    // 랜덤 샘플링 구현 (현재는 순차적으로 처음 batch_size개를 사용)
    let mut rng = rand::thread_rng();
    
    // 인덱스 벡터 생성
    let mut indices: Vec<usize> = (0..buffer.len()).collect();
    
    // Fisher-Yates 알고리즘으로 인덱스 섞기
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    
    // 섞인 인덱스에서 batch_size개 선택
    for &idx in indices.iter().take(batch_size as usize) {
        let exp = &buffer.buffer[idx];
        states.extend_from_slice(&exp.state);
        actions.extend_from_slice(&exp.action);
        rewards.push(exp.reward);
        next_states.extend_from_slice(&exp.next_state);
        dones.push(if exp.done { 1.0 } else { 0.0 });
        log_probs.push(exp.log_prob);
    }
    
    // 모든 Tensor::of_slice를 Tensor::f_from_slice로 변경
    let states_result = Tensor::f_from_slice(&states)
        .map(|t| t.view([-1, buffer.buffer[0].state.len() as i64]).to_device(device));
        
    let actions_result = Tensor::f_from_slice(&actions)
        .map(|t| t.view([-1, buffer.buffer[0].action.len() as i64]).to_device(device));
        
    let rewards_result = Tensor::f_from_slice(&rewards)
        .map(|t| t.to_device(device));
        
    let next_states_result = Tensor::f_from_slice(&next_states)
        .map(|t| t.view([-1, buffer.buffer[0].next_state.len() as i64]).to_device(device));
        
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

// 정책 액션 평가 함수
pub fn evaluate_action_fixed(actor: &nn::Sequential, states: &Tensor, actions: &Tensor) -> (Tensor, Tensor) {
    let mean = actor.forward(states);
    let std = Tensor::f_from_slice(&[0.5f32; 4]).unwrap()
        .to_device(states.device())
        .expand(&[states.size()[0], 4], false);
    
    // 텐서 연산에 필요한 상수들
    let dim: &[i64] = &[-1];
    let two_tensor = Tensor::f_from_slice(&[2.0f32]).unwrap().to_device(states.device());
    let pi_const = Tensor::f_from_slice(&[2.0 * std::f32::consts::PI]).unwrap().log() / 2.0;
    
    // 로그 확률 계산
    let scaled_diff = (actions - &mean) / &std;
    let log_prob = -(&scaled_diff.pow(&two_tensor) / 2.0 + &std.log() + pi_const)
        .sum_dim_intlist(dim, false, Kind::Float);
    
    // 엔트로피 계산
    let point_five = Tensor::f_from_slice(&[0.5f32]).unwrap().to_device(states.device());
    let pi_const_clone = Tensor::f_from_slice(&[2.0 * std::f32::consts::PI]).unwrap().log() / 2.0;
    let entropy = (&point_five + &point_five * &pi_const_clone * 2.0 + &std.log())
        .sum_dim_intlist(dim, false, Kind::Float);
    
    (log_prob, entropy)
}

// GAE(Generalized Advantage Estimation) 계산 - CPU 벡터 기반 계산
pub fn compute_gae_fixed(rewards: &Tensor, values: &Tensor, next_values: &Tensor, dones: &Tensor, gamma: f64, lambda: f64) -> Tensor {
    // 원래 디바이스 저장
    let device = rewards.device();
    
    // 스칼라 값을 f32로 변환
    let gamma_t = gamma as f32;
    let lambda_t = lambda as f32;
    
    // CPU로 데이터 이동
    let rewards_cpu = rewards.to_device(Device::Cpu);
    let values_cpu = values.to_device(Device::Cpu);
    let next_values_cpu = next_values.to_device(Device::Cpu);
    let dones_cpu = dones.to_device(Device::Cpu);
    
    // 배치 크기 계산
    let batch_size = rewards_cpu.size()[0] as usize;
    
    // CPU에서 데이터 추출
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
    
    // CPU에서 GAE 계산
    let mut advantages = vec![0.0f32; batch_size];
    let mut last_gae = 0.0;
    
    for i in (0..batch_size).rev() {
        let delta = rewards_vec[i] + gamma_t * next_values_vec[i] * (1.0 - dones_vec[i]) - values_vec[i];
        last_gae = delta + gamma_t * lambda_t * (1.0 - dones_vec[i]) * last_gae;
        advantages[i] = last_gae;
    }
    
    // 결과를 텐서로 변환하고 원래 디바이스로 복귀
    match Tensor::f_from_slice(&advantages) {
        Ok(tensor) => tensor.to_device(device),
        Err(_) => Tensor::zeros(&[batch_size as i64], (Kind::Float, device))
    }
}

// Patch for ppo_update_system fixing the value loss calculation
pub fn value_loss_fixed(value_pred: &Tensor, returns: &Tensor) -> Tensor {
    let two_tensor = Tensor::f_from_slice(&[2.0f32]).unwrap().to_device(value_pred.device());
    (value_pred - returns).pow(&two_tensor).mean(Kind::Float)
} 