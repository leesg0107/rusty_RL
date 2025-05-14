use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;
use crate::rl_patch::{RLThread, PPOParams, Experience, AgentMemory, add_experience, request_action, get_action_result, set_training_enabled, update_reward_info, request_save_model, start_training_thread, get_training_stats};
use crate::agent::DroneAgent;
use crate::environment::{Environment, Target};

// main.rs에서 정의된 ControlMode를 가져옴
use crate::ControlMode;

/// 강화학습 시스템을 위한 Bevy 플러그인
pub struct RLPlugin;

impl Plugin for RLPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RLThread>()
           .init_resource::<PPOParams>()
           .add_systems(Startup, setup_rl_thread)
           .add_systems(Update, (
               agent_state_system
                   .run_if(|mode: Res<ControlMode>| matches!(*mode, ControlMode::ReinforcementLearning)),
               agent_action_system
                   .run_if(|mode: Res<ControlMode>| matches!(*mode, ControlMode::ReinforcementLearning)),
               check_learning_status_system,
               handle_keyboard_controls
           ));
    }
}

// RL 스레드 설정 시스템
fn setup_rl_thread(mut rl_thread: ResMut<RLThread>, params: Res<PPOParams>) {
    println!("강화학습 스레드 설정 중...");
    start_training_thread(&mut rl_thread, params.clone());
    // 학습 자동 활성화
    set_training_enabled(&rl_thread, true);
    println!("강화학습 스레드 시작 완료 (학습 활성화됨)");
}

// 에이전트 상태 수집 시스템
fn agent_state_system(
    mut query: Query<(&DroneAgent, &Transform, &mut AgentMemory)>,
    target_query: Query<&Transform, With<Target>>,
    rl_thread: Res<RLThread>,
    _env: Res<Environment>,
    time: Res<Time>,
) {
    // 목표 위치 가져오기
    let target_pos = if let Ok(target_transform) = target_query.get_single() {
        target_transform.translation
    } else {
        return; // 목표가 없으면 종료
    };
    
    // 정기적으로 통계 출력 (5초마다)
    let should_print_stats = (time.elapsed_seconds() % 5.0) < time.delta_seconds();
    
    for (agent, transform, mut memory) in query.iter_mut() {
        // 현재 상태 추출
        let current_state = extract_state(agent, transform, target_pos);
        
        // 이전 상태가 있으면 경험 저장
        if let (Some(prev_state), Some(prev_action), Some(prev_log_prob), Some(prev_value)) = 
            (memory.prev_state.clone(), memory.prev_action.clone(), memory.prev_log_prob, memory.prev_value) {
            
            // 클론한 값을 사용하여 메모리 가변 대출 문제 해결
            let reward = agent.reward;
            let done = agent.done;
            
            // 보상 누적
            memory.cumulative_reward += reward;
            
            // 경험 생성
            let experience = Experience {
                state: prev_state,
                action: prev_action,
                reward,
                next_state: current_state.clone(),
                done,
                log_prob: prev_log_prob,
                value: prev_value,
            };
            
            // 스레드 공유 버퍼에 추가
            add_experience(&rl_thread, experience);
        }
        
        // 정기적으로 현재 에피소드 통계 출력
        if should_print_stats {
            let (training_steps, best_reward, avg_reward) = get_training_stats(&rl_thread);
            
            println!(
                "=== 현재 에피소드 정보 ===\n\
                 에피소드: {}, 스텝: {}, 누적 보상: {:.2}\n\
                 최고 보상: {:.2}, 평균 보상: {:.2}, 학습 단계: {}", 
                memory.total_episodes + 1, memory.episode_steps, memory.cumulative_reward,
                best_reward, avg_reward, training_steps
            );
        }
        
        // 에피소드 종료 처리
        if agent.done {
            // 에피소드 통계 업데이트
            memory.total_episodes += 1;
            
            // 복사본을 만들어서 사용
            let cumulative_reward = memory.cumulative_reward;
            memory.rewards_history.push(cumulative_reward);
            
            // 리워드 정보 업데이트
            update_reward_info(&rl_thread, cumulative_reward);
            
            if cumulative_reward > memory.best_reward {
                memory.best_reward = cumulative_reward;
            }
            
            // 에피소드 정보 출력 (항상 출력)
            let (training_steps, best_reward, avg_reward) = get_training_stats(&rl_thread);
            
            println!("=== 에피소드 {} 종료 ===", memory.total_episodes);
            println!("스텝: {}, 누적 보상: {:.2}", memory.episode_steps, cumulative_reward);
            println!("최고 보상: {:.2}, 평균 보상: {:.2}", best_reward, avg_reward);
            println!("학습 단계: {}", training_steps);
            println!("==================");
            
            // 메모리 초기화
            memory.prev_state = None;
            memory.prev_action = None;
            memory.prev_log_prob = None;
            memory.prev_value = None;
            memory.cumulative_reward = 0.0;
            memory.episode_steps = 0;
            
            return;
        }
        
        // RL 스레드에 액션 요청
        request_action(&rl_thread, current_state.clone());
        
        // 현재 상태 저장
        memory.prev_state = Some(current_state);
    }
}

// 에이전트 액션 적용 시스템
fn agent_action_system(
    mut query: Query<(&mut DroneAgent, &mut AgentMemory, &Transform, &Velocity)>,
    rl_thread: Res<RLThread>,
    time: Res<Time>,
) {
    // 매 초마다 실행 여부 확인 로그 출력
    if (time.elapsed_seconds() % 5.0) < time.delta_seconds() {
        println!("RL 액션 시스템 실행 중... 에이전트 수: {}", query.iter().count());
    }

    for (mut agent, mut memory, transform, velocity) in query.iter_mut() {
        // 스레드로부터 액션 결과 가져오기
        let action_result = get_action_result(&rl_thread);
        
        // 결과 여부에 따라 로그 출력
        if action_result.is_none() && (time.elapsed_seconds() % 5.0) < time.delta_seconds() {
            println!("RL 스레드로부터 액션을 받지 못했습니다.");
            println!("현재 위치: {:?}, 속도: {:?}", transform.translation, velocity.linvel);
            // 스레드에 현재 상태 기반 액션 재요청
            if let Some(state) = &memory.prev_state {
                request_action(&rl_thread, state.clone());
                println!("액션 재요청함");
            }
            continue;
        }
        
        if let Some((action, value, log_prob)) = action_result {
            // 액션을 드론 제어 입력으로 변환 (안전 제약 추가)
            let (thrust, torque) = action_to_control_safe(&action, transform, velocity);
            
            // 드론에 제어 입력 적용
            agent.thrust = thrust;
            agent.torque = torque;
            
            // 로깅 빈도 증가
            if (time.elapsed_seconds() % 3.0) < time.delta_seconds() {
                println!(
                    "RL 액션 적용: 추력={:.2}, 토크={:?}, 위치={:?}, 속도={:?}",
                    thrust, torque, transform.translation, velocity.linvel
                );
                println!("원시 액션 값: {:?}, 가치: {:.3}, 로그확률: {:.3}", 
                         &action, value, log_prob);
            }
            
            // 현재 액션과 정보 저장
            memory.prev_action = Some(action);
            memory.prev_log_prob = Some(log_prob);
            memory.prev_value = Some(value);
            
            // 스텝 카운터 증가
            memory.episode_steps += 1;
            memory.total_steps += 1;
            
            // 에이전트의 목표 이동 방향도 설정 (드론 제어 시스템과 통합)
            // 전방 벡터를 목표 이동 방향으로 설정
            let forward = transform.forward();
            let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            
            // 토크 방향에 따라 목표 이동 방향 설정
            let pitch = agent.torque.x;  // 앞/뒤 기울기
            let roll = agent.torque.z;   // 좌/우 기울기
            
            // 토크의 방향에 따라 이동 방향 설정
            let mut move_dir = Vec3::ZERO;
            if pitch.abs() > 0.01 {
                move_dir += forward * pitch.signum();
            }
            
            if roll.abs() > 0.01 {
                move_dir += transform.right() * -roll.signum();
            }
            
            // 정규화 및 적용
            if move_dir.length_squared() > 0.01 {
                agent.target_direction = move_dir.normalize();
                agent.target_speed = (pitch.powi(2) + roll.powi(2)).sqrt() * 5.0;
            } else {
                agent.target_direction = Vec3::ZERO;
                agent.target_speed = 0.0;
            }
        }
    }
}

// 안전한 행동 변환 (극단적인 행동 제한)
fn action_to_control_safe(action: &[f32], transform: &Transform, velocity: &Velocity) -> (f32, Vec3) {
    // 기본 행동 변환
    let mut thrust = action[0].clamp(-1.0, 1.0) * 0.5 + 0.5; // 0.0 ~ 1.0 범위로 변환
    let mut torque = Vec3::new(
        action[1].clamp(-1.0, 1.0),
        action[2].clamp(-1.0, 1.0),
        action[3].clamp(-1.0, 1.0),
    );
    
    // 안전 제약 추가
    
    // 1. 높이가 낮을 때는 최소 추력 보장 (바닥 충돌 방지)
    if transform.translation.y < 1.0 {
        // 높이가 낮을수록 더 강한 최소 추력 적용
        let min_thrust = 0.6 + (1.0 - transform.translation.y) * 0.3;
        thrust = thrust.max(min_thrust);
        
        // 하강 속도가 빠를 때도 추력 증가
        if velocity.linvel.y < -1.0 {
            thrust = thrust.max(0.7 - velocity.linvel.y * 0.05);
        }
    }
    
    // 2. 높이가 높을 때는 최대 추력 제한 (무한 상승 방지)
    if transform.translation.y > 15.0 {
        // 높이가 높을수록 더 강한 최대 추력 제한
        let max_thrust = 0.6 - (transform.translation.y - 15.0) * 0.02;
        thrust = thrust.min(max_thrust);
    }
    
    // 3. 속도가 매우 빠를 때는 토크 제한 (과도한 회전 방지)
    let speed = velocity.linvel.length();
    if speed > 5.0 {
        // 속도에 비례하여 토크 크기 감소
        let torque_scale = 1.0 - ((speed - 5.0) * 0.1).min(0.7);
        torque *= torque_scale;
    }
    
    // 4. 각속도가 매우 빠를 때는 토크 감소 (과도한 회전 방지)
    let angular_speed = velocity.angvel.length();
    if angular_speed > 2.0 {
        // 각속도에 비례하여 토크 크기 감소
        let torque_scale = 1.0 - ((angular_speed - 2.0) * 0.2).min(0.8);
        torque *= torque_scale;
    }
    
    (thrust, torque)
}

// 학습 상태 확인 시스템
fn check_learning_status_system(
    time: Res<Time>,
    rl_thread: ResMut<RLThread>,
) {
    // 5초마다 학습 상태 출력
    if (time.elapsed_seconds() % 5.0) < 0.05 {
        let (training_steps, best_reward, avg_reward) = get_training_stats(&rl_thread);
        if training_steps > 0 {
            debug!("학습 상태: 단계={}, 최고={:.2}, 평균={:.2}", training_steps, best_reward, avg_reward);
        }
    }
}

// 키보드 제어 시스템
fn handle_keyboard_controls(
    keyboard_input: Res<Input<KeyCode>>,
    rl_thread: Res<RLThread>,
) {
    // T키: 학습 활성화/비활성화
    if keyboard_input.just_pressed(KeyCode::T) {
        let is_training = {
            if let Ok(data) = rl_thread.shared_data.lock() {
                data.is_training
            } else {
                false
            }
        };
        
        set_training_enabled(&rl_thread, !is_training);
        info!("학습 {}", if !is_training { "활성화됨" } else { "비활성화됨" });
    }
    
    // S키: 모델 저장
    if keyboard_input.just_pressed(KeyCode::S) {
        request_save_model(&rl_thread);
        info!("모델 저장 요청됨");
    }
}

// 상태 추출 함수
fn extract_state(agent: &DroneAgent, transform: &Transform, target_pos: Vec3) -> Vec<f32> {
    let pos = transform.translation;
    let vel = agent.velocity;
    let target_dir = target_pos - pos;
    
    vec![
        pos.x, pos.y, pos.z,           // 드론 위치
        vel.x, vel.y, vel.z,           // 드론 속도
        target_dir.x, target_dir.y, target_dir.z, // 목표까지 방향
    ]
}

// 행동을 드론 제어 입력으로 변환 (기존 함수는 유지)
fn action_to_control(action: &[f32]) -> (f32, Vec3) {
    let thrust = action[0].clamp(-1.0, 1.0) * 0.5 + 0.5; // 0.0 ~ 1.0 범위로 변환
    let torque = Vec3::new(
        action[1].clamp(-1.0, 1.0),
        action[2].clamp(-1.0, 1.0),
        action[3].clamp(-1.0, 1.0),
    );
    
    (thrust, torque)
} 