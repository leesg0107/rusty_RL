use bevy::prelude::*;
use rand::Rng;
use bevy_rapier3d::prelude::*;
use crate::agent::DroneAgent;

#[derive(Resource)]
pub struct Environment {
    pub episode: usize,
    pub step: usize,
    pub max_steps: usize,
    pub target_position: Vec3,
    pub stabilization_period: usize, // 안정화 기간 (스텝 수)
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            episode: 0,
            step: 0,
            max_steps: 1000,
            target_position: Vec3::new(0.0, 5.0, 0.0),
            stabilization_period: 50, // 처음 50스텝 동안은 안정화 기간
        }
    }
}

// 환경 초기화 시스템
pub fn setup_environment(mut commands: Commands) {
    commands.insert_resource(Environment::default());
}

// 목표물 식별을 위한 컴포넌트
#[derive(Component)]
pub struct Target;

// 목표 지점 생성 시스템
pub fn spawn_target(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position: Vec3,
) {
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 0.25,
                sectors: 32,
                stacks: 16,
            })),
            material: materials.add(Color::rgb(0.9, 0.9, 0.1).into()),
            transform: Transform::from_translation(position),
            ..default()
        },
        Name::new("Target"),
        Target, // Target 컴포넌트 추가
    ));
}

// 보상 계산 시스템
pub fn calculate_reward(
    mut query: Query<(&mut DroneAgent, &Transform, &Velocity)>,
    target_query: Query<&Transform, With<Target>>,
    time: Res<Time>,
) {
    // 목표 위치 가져오기
    let target_pos = if let Ok(target_transform) = target_query.get_single() {
        target_transform.translation
    } else {
        return; // 목표가 없으면 종료
    };
    
    // 디버그 출력용 변수 (1초마다 출력)
    let should_debug = (time.elapsed_seconds() % 1.0) < time.delta_seconds();
    
    for (mut agent, transform, velocity) in query.iter_mut() {
        let pos = transform.translation;
        
        // 목표까지의 거리
        let distance = (pos - target_pos).length();
        
        // 기본 보상: 거리에 반비례 (약화)
        let distance_reward = -0.01 * distance; // 0.03 -> 0.01으로 더 약화
        
        // 속도 페널티: 과도한 속도 억제 (약화)
        let velocity_penalty = -0.01 * velocity.linvel.length_squared(); // 0.02 -> 0.01로 더 약화
        
        // 에너지 효율성: 추력과 토크 사용 최소화 (약화)
        let energy_penalty = -0.001 * (agent.thrust.powi(2) + agent.torque.length_squared()); // 0.002 -> 0.001로 더 약화
        
        // 목표 도달 보상 (유지)
        let goal_reward = if distance < 1.0 {
            10.0 * (1.0 - distance)
        } else {
            0.0
        };
        
        // 충돌 페널티 (조정)
        let collision_penalty = if pos.y < 0.3 {
            -5.0 // -20.0 -> -5.0로 대폭 약화 (즉시 종료되지 않도록)
        } else {
            0.0
        };
        
        // 안정성 보상: 수평 자세 유지 (강화)
        let up_vector = transform.rotation * Vec3::Y;
        let stability_reward = 2.5 * up_vector.dot(Vec3::Y); // 1.5 -> 2.5로 더 강화
        
        // 높이 유지 보상 (단순화)
        let height_reward = if pos.y < 0.3 {
            0.0 // 너무 낮으면 보상 없음
        } else if pos.y < 1.0 {
            // 낮은 높이: 서서히 증가
            3.0 * pos.y
        } else if pos.y < 6.0 {
            // 적정 높이: 높은 보상
            3.0
        } else {
            // 너무 높으면 감소
            3.0 - 0.2 * (pos.y - 6.0)
        };
        
        // 생존 보상 (강화)
        let survival_reward = 0.3; // 0.2 -> 0.3으로 강화
        
        // 호버링 보상: 수직 속도가 낮을 때 추가 보상
        let hovering_reward = if velocity.linvel.length() < 1.0 && pos.y > 0.5 {
            2.0 * (1.0 - velocity.linvel.length()) // 1.0 -> 2.0으로 증가, 모든 방향 속도 고려
        } else {
            0.0
        };
        
        // 향상된 자세 안정성 보상
        let angular_velocity_penalty = -0.5 * velocity.angvel.length_squared(); // 회전 속도 페널티 추가
        
        // 경계 이탈 감지 (종료 대신 보상 조정)
        let boundary_penalty = if pos.y > 20.0 || pos.x.abs() > 20.0 || pos.z.abs() > 20.0 {
            // 경계 이탈시 큰 페널티 부여
            -5.0
        } else if pos.y > 15.0 || pos.x.abs() > 15.0 || pos.z.abs() > 15.0 {
            // 경계에 가까워지면 점진적인 페널티 부여
            -2.0 * (1.0 + (pos.y.max(pos.x.abs()).max(pos.z.abs()) - 15.0) / 5.0)
        } else {
            0.0
        };
        
        // 총 보상
        agent.reward = distance_reward + velocity_penalty + energy_penalty + 
                       goal_reward + collision_penalty + stability_reward + 
                       height_reward + survival_reward + hovering_reward + 
                       angular_velocity_penalty + boundary_penalty;
        
        // 1초마다 보상 세부 내역 출력
        if should_debug {
            println!(
                "==보상 세부 내역== 위치: {:.1}, 높이: {:.1}, 거리: {:.1} \n \
                 거리: {:.2}, 속도: {:.2}, 에너지: {:.2}, 목표: {:.2}, \n \
                 충돌: {:.2}, 안정성: {:.2}, 높이: {:.2}, 생존: {:.2}, 호버링: {:.2}, 회전안정: {:.2}, 경계: {:.2} \n \
                 => 총 보상: {:.2}",
                pos.x, pos.y, distance,
                distance_reward, velocity_penalty, energy_penalty, goal_reward,
                collision_penalty, stability_reward, height_reward, survival_reward, hovering_reward, 
                angular_velocity_penalty, boundary_penalty,
                agent.reward
            );
        }
        
        // 에피소드 종료 조건 대폭 완화
        // 에피소드 종료 조건을 경계 훨씬 밖으로 확장하여 에이전트가 쉽게 사라지지 않도록 함
        agent.done = distance < 0.5  // 목표 도달 시 (성공)
               || pos.y < 0.1        // 지면과 충돌했을 때 (실패)
               || pos.y > 30.0       // 매우 높이 올라갔을 때 (실패)
               || pos.x.abs() > 30.0 // 수평으로 매우 멀리 갔을 때 (실패)
               || pos.z.abs() > 30.0;// 수평으로 매우 멀리 갔을 때 (실패)
        
        // 목표 도달 시 추가 보상
        if distance < 0.5 {
            agent.reward += 100.0; // 50.0 -> 100.0
            println!("목표 도달! 거리: {:.2}, 최종 보상: {:.2}", distance, agent.reward);
        }
        
        // 충돌 시 페널티 (약화)
        if pos.y < 0.1 {  // 0.2 -> 0.1로 임계값 낮춤
            agent.reward -= 5.0; // 10.0 -> 5.0으로 추가 약화
            println!("충돌 발생! 높이: {:.2}, 최종 보상: {:.2}", pos.y, agent.reward);
        }
        
        // 경계 이탈 시 페널티 (약화)
        if pos.y > 30.0 || pos.x.abs() > 30.0 || pos.z.abs() > 30.0 {
            agent.reward -= 5.0; // 10.0 -> 5.0으로 추가 약화
            println!("경계 이탈! 위치: {:?}, 최종 보상: {:.2}", pos, agent.reward);
        }
    }
}

// 새로운 경계 안전장치 시스템 추가
pub fn enforce_boundary_safety(
    mut query: Query<(&mut Transform, &mut Velocity)>,
    time: Res<Time>,
) {
    // 경계 구성
    const MAX_HEIGHT: f32 = 20.0;
    const MAX_HORIZONTAL: f32 = 20.0;
    const SAFETY_MARGIN: f32 = 2.0;
    
    for (mut transform, mut velocity) in query.iter_mut() {
        let pos = transform.translation;
        let mut should_correct = false;
        let mut new_pos = pos;
        
        // 높이 제한
        if pos.y > MAX_HEIGHT {
            new_pos.y = MAX_HEIGHT - SAFETY_MARGIN;
            should_correct = true;
        } else if pos.y < 0.5 {
            // 너무 낮을 때는 안전 높이로 살짝 올림
            new_pos.y = 0.5 + SAFETY_MARGIN;
            should_correct = true;
        }
        
        // 수평 제한 (X축)
        if pos.x.abs() > MAX_HORIZONTAL {
            new_pos.x = if pos.x > 0.0 {
                MAX_HORIZONTAL - SAFETY_MARGIN
            } else {
                -MAX_HORIZONTAL + SAFETY_MARGIN
            };
            should_correct = true;
        }
        
        // 수평 제한 (Z축)
        if pos.z.abs() > MAX_HORIZONTAL {
            new_pos.z = if pos.z > 0.0 {
                MAX_HORIZONTAL - SAFETY_MARGIN
            } else {
                -MAX_HORIZONTAL + SAFETY_MARGIN
            };
            should_correct = true;
        }
        
        // 위치 조정이 필요하면 적용
        if should_correct {
            // 위치 조정 사실을 로그로 남김
            println!(
                "경계 안전장치 작동: 위치 [{:.1}, {:.1}, {:.1}] -> [{:.1}, {:.1}, {:.1}]",
                pos.x, pos.y, pos.z, new_pos.x, new_pos.y, new_pos.z
            );
            
            // 새 위치로 이동
            transform.translation = new_pos;
            
            // 속도 감소 (안전하게 정지하도록)
            velocity.linvel *= 0.2;
            velocity.angvel *= 0.2;
        }
    }
}

// 환경 리셋 시스템 (이 함수는 main.rs에서 직접 구현)
pub fn reset_environment(
    _commands: Commands,
    mut env: ResMut<Environment>,
    mut query: Query<(Entity, &mut Transform, &mut DroneAgent)>,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    // 에피소드 증가
    env.episode += 1;
    env.step = 0;
    
    // 새로운 목표 위치 설정
    let mut rng = rand::thread_rng();
    env.target_position = Vec3::new(
        rng.gen_range(-10.0..10.0),
        rng.gen_range(3.0..8.0),
        rng.gen_range(-10.0..10.0),
    );
    
    // 에이전트 리셋
    for (_entity, mut transform, mut agent) in query.iter_mut() {
        transform.translation = Vec3::new(0.0, 1.0, 0.0);
        transform.rotation = Quat::IDENTITY;
        
        agent.position = transform.translation;
        agent.velocity = Vec3::ZERO;
        agent.rotation = Quat::IDENTITY;
        agent.reward = 0.0;
        agent.done = false;
        agent.thrust = 0.0;
        agent.torque = Vec3::ZERO;
    }
} 