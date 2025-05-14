use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_egui::EguiPlugin;
use rand;
use rand::Rng;

mod agent;
mod environment;
// 이전 rl.rs 모듈을 제거하지는 않지만 불러오지 않음
// mod rl;
mod rl_patch;
mod rl_plugin;

use agent::{DroneAgent, control_drone, spawn_keyboard_controlled_drone, keyboard_control_system, KeyboardControlled, ControlGains, HoverSettings};
use environment::{Environment, spawn_target, calculate_reward, Target, enforce_boundary_safety};
// rl_patch에서 모든 필요한 구조체를 가져옴
use rl_patch::{PPOParams, RLThread, AgentMemory, request_save_model, request_terminate};
use rl_plugin::RLPlugin;

// 제어 모드 리소스
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    Keyboard,
    ReinforcementLearning,
    CircleTest, // 원형 비행 테스트 모드 추가
}

impl Default for ControlMode {
    fn default() -> Self {
        ControlMode::ReinforcementLearning // 기본값을 RL 모드로 변경
    }
}

// 원형 비행 파라미터
#[derive(Resource)]
struct CircleFlightParams {
    radius: f32,         // 원의 반지름
    height: f32,         // 비행 높이
    speed: f32,          // 회전 속도 (라디안/초)
    elapsed_time: f32,   // 경과 시간
    center: Vec3,        // 원의 중심
    initialized: bool,   // 초기화 여부
}

impl Default for CircleFlightParams {
    fn default() -> Self {
        Self {
            radius: 5.0,
            height: 5.0,
            speed: 0.3,  // 0.5 -> 0.3로 감소 (더 천천히 회전)
            elapsed_time: 0.0,
            center: Vec3::new(0.0, 0.0, 0.0),
            initialized: false,
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(EguiPlugin)
        .add_plugins(RLPlugin)
        .insert_resource(PPOParams::default())
        .insert_resource(ControlMode::default())
        .insert_resource(ControlGains::default())
        .insert_resource(HoverSettings::default())  // 호버링 설정 리소스 추가
        .insert_resource(Environment::default())
        .insert_resource(CircleFlightParams::default()) // 원형 비행 파라미터 리소스 추가
        .add_systems(Startup, setup_test_environment)
        .add_systems(Update, toggle_control_mode)
        .add_systems(Update, keyboard_control_system
            .run_if(|mode: Res<ControlMode>| matches!(*mode, ControlMode::Keyboard)))
        .add_systems(Update, keyboard_reset_system)
        .add_systems(Update, control_drone)  // 모든 모드에서 항상 실행됨
        .add_systems(Update, enforce_boundary_safety) // 새로운 경계 안전장치 추가
        .add_systems(Update, circle_flight_system
            .run_if(|mode: Res<ControlMode>| matches!(*mode, ControlMode::CircleTest)))
        .add_systems(Update, calculate_reward)
        .add_systems(Update, print_drone_info)
        .add_systems(Update, test_keyboard_input)
        .add_systems(Update, check_reset
            .run_if(|mode: Res<ControlMode>| matches!(*mode, ControlMode::ReinforcementLearning)))
        .add_systems(Update, update_learning_stats_ui)
        .add_systems(Last, cleanup_rl_thread)
        .run();
}

// RL 스레드 정리 시스템
fn cleanup_rl_thread(rl_thread: Res<RLThread>) {
    request_terminate(&rl_thread);
}

// 테스트 환경 설정
fn setup_test_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mode: Res<ControlMode>,
    mut hover_settings: ResMut<HoverSettings>,
) {
    commands.insert_resource(Environment::default());
    
    // 초기 호버링 설정 - 시작 높이를 2미터로 설정
    hover_settings.hover_height = 2.0;
    hover_settings.height_pid.reset();
    
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane { size: 50.0, subdivisions: 0 })),
            material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
            ..default()
        },
        Collider::cuboid(25.0, 0.1, 25.0),
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 10.0, 0.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-10.0, 10.0, 10.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // 드론 초기 높이를 2미터로 낮춤 (테스트 용이성 향상)
    let drone_entity = spawn_keyboard_controlled_drone(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, 2.0, 0.0));
    
    // 드론이 강화학습 모드에서 사용할 수 있도록 AgentMemory 컴포넌트 추가
    if matches!(*mode, ControlMode::ReinforcementLearning) {
        commands.entity(drone_entity).insert(AgentMemory::default());
        println!("초기 드론에 AgentMemory 컴포넌트가 추가되었습니다.");
    }
    
    // 타겟을 5미터 높이에 배치 (상승 유도)
    spawn_target(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, 5.0, 0.0));
    
    spawn_mode_text(&mut commands, *mode);
    
    println!("=== 드론 키보드 제어 안내 ===");
    println!("상승: 스페이스바");
    println!("하강: 좌측 Shift");
    println!("전진/후진: 위/아래 화살표");
    println!("좌/우 이동: 좌/우 화살표");
    println!("회전: A/D 키");
    println!("리셋: R 키");
    println!("모드 전환: Tab");
    println!("학습 시작/정지: T");
    println!("모델 저장: S");
    println!("초기 호버링 높이: {:.1}m", hover_settings.hover_height);
}

// 드론 정보 출력 시스템
fn print_drone_info(
    query: Query<(&Transform, &Velocity, &DroneAgent), With<KeyboardControlled>>,
    time: Res<Time>,
    mode: Res<ControlMode>,
) {
    if (time.elapsed_seconds() % 1.0) < time.delta_seconds() {
        for (transform, velocity, agent) in query.iter() {
            println!("모드: {:?}", *mode);
            println!("위치: {:?}, 속도: {:?}, 높이: {:.2}", 
                     transform.translation, velocity.linvel, transform.translation.y);
            println!("추력: {:.2}, 토크: {:?}", agent.thrust, agent.torque);
            println!("---------------------");
        }
    }
}

// 환경 리셋 확인 시스템
fn check_reset(
    mut commands: Commands,
    mut env: ResMut<Environment>,
    query: Query<(Entity, &Transform, &DroneAgent)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target_query: Query<(Entity, &Transform), With<Target>>,
    mode: Res<ControlMode>,
    mut hover_settings: ResMut<HoverSettings>,
    time: Res<Time>,
) {
    if !matches!(*mode, ControlMode::ReinforcementLearning) {
        return;
    }
    
    let mut should_reset = false;
    let mut entities_to_despawn = Vec::new();
    let mut has_valid_drone = false;
    let mut drone_pos = Vec3::ZERO;
    let mut reset_reason = "기본 리셋";
    
    // 목표 위치 확인
    let target_pos = if let Ok((_, target_transform)) = target_query.get_single() {
        target_transform.translation
    } else {
        Vec3::new(0.0, 5.0, 0.0) // 기본 목표 위치
    };
    
    // 드론 에이전트 확인
    for (entity, transform, agent) in query.iter() {
        has_valid_drone = true;
        drone_pos = transform.translation;
        
        // 종료 조건 확인
        if agent.done || env.step >= env.max_steps {
            should_reset = true;
            entities_to_despawn.push(entity);
            
            // 에피소드 종료 사유 로깅
            if agent.done {
                reset_reason = "에이전트 완료";
            } else {
                reset_reason = "최대 스텝 도달";
            }
        }
        
        // 환경 경계 체크 - 드론이 허용 범위를 벗어난 경우
        let boundary = 20.0; // 허용 경계 범위
        let min_height = 0.1; // 최소 높이
        let max_height = 20.0; // 최대 높이
        
        if transform.translation.x.abs() > boundary || 
           transform.translation.z.abs() > boundary ||
           transform.translation.y < min_height ||
           transform.translation.y > max_height {
            should_reset = true;
            entities_to_despawn.push(entity);
            reset_reason = "경계 이탈";
            println!("드론이 경계를 벗어났습니다: 위치 = {:?}", transform.translation);
        }
        
        // 정지 상태 확인 - 드론이 너무 오랫동안 목표에 접근하지 못함
        // 목표까지 거리 계산
        let distance_to_target = (transform.translation - target_pos).length();
        
        // 오랜 시간 동안 움직이지 않거나 목표에 접근하지 못하는 경우
        if env.step > 500 && distance_to_target > 8.0 {
            should_reset = true;
            entities_to_despawn.push(entity);
            reset_reason = "오랜 시간 목표 미달성";
            println!("드론이 오랜 시간({} 스텝) 목표에 접근하지 못했습니다. 거리: {:.2}", env.step, distance_to_target);
        }
        
        // 속도가 매우 낮은 상태로 오래 있는 경우도 리셋 (드론이 멈춰있는 경우)
        if env.step > 300 {
            let velocity = agent.velocity.length();
            if velocity < 0.1 && distance_to_target > 5.0 {
                should_reset = true; 
                entities_to_despawn.push(entity);
                reset_reason = "드론 정지 상태 감지";
                println!("드론이 멈춰있습니다. 속도: {:.2}, 목표까지 거리: {:.2}", velocity, distance_to_target);
            }
        }
    }
    
    // 에이전트가 없는 경우 강제 리셋 (보호 장치)
    if !has_valid_drone {
        println!("!!! 경고: 유효한 드론 에이전트가 없습니다. 환경을 강제로 리셋합니다 !!!");
        should_reset = true;
        reset_reason = "유효한 드론 없음";
    }
    
    if should_reset {
        env.episode += 1;
        env.step = 0;
        
        // 기존 에이전트 삭제
        for entity in entities_to_despawn {
            println!("에이전트 삭제");
            commands.entity(entity).despawn_recursive();
        }
        
        // 타겟 삭제
        for (entity, _) in target_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
        
        // 호버링 높이를 2.0으로 리셋
        hover_settings.hover_height = 2.0;
        hover_settings.height_pid.reset();
        
        // 드론 스폰 위치 결정 - 에이전트가 없거나 경계 밖이면 중앙에 스폰
        let spawn_pos = if has_valid_drone && 
                          drone_pos.y > 0.5 && drone_pos.y < 15.0 && 
                          drone_pos.x.abs() < 15.0 && drone_pos.z.abs() < 15.0 {
            // 기존 드론 위치가 유효하면 해당 위치 근처에 스폰 (약간 위로)
            Vec3::new(drone_pos.x, drone_pos.y + 1.0, drone_pos.z)
        } else {
            // 그 외의 경우 기본 스폰 위치
            Vec3::new(0.0, 2.0, 0.0)
        };
        
        println!("새 드론 생성: 위치 = {:?}", spawn_pos);
        let drone_entity = spawn_keyboard_controlled_drone(&mut commands, &mut meshes, &mut materials, spawn_pos);
        
        // AgentMemory 컴포넌트 추가 - 강화학습에 필수
        commands.entity(drone_entity).insert(AgentMemory::default());
        println!("강화학습 에이전트 메모리 컴포넌트 추가됨");
        
        // 타겟 위치 설정 (초기 학습은 쉽게, 나중에 점점 어렵게)
        let target_pos = if env.episode < 10 {
            // 첫 10 에피소드: 고정 위치 (쉬움)
            Vec3::new(0.0, 5.0, 0.0)
        } else if env.episode < 20 {
            // 다음 10 에피소드: 약간 랜덤 (중간)
            let mut rng = rand::thread_rng();
            Vec3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(3.0..6.0),
                rng.gen_range(-2.0..2.0),
            )
        } else {
            // 이후: 더 넓은 범위 (어려움)
            let mut rng = rand::thread_rng();
            Vec3::new(
                rng.gen_range(-5.0..5.0),
                rng.gen_range(3.0..6.0),
                rng.gen_range(-5.0..5.0),
            )
        };
        
        spawn_target(&mut commands, &mut meshes, &mut materials, target_pos);
        
        println!("=== 에피소드 {} 시작 === 타겟 위치: {:?}, 리셋 사유: {}", env.episode, target_pos, reset_reason);
    }
    
    // 스텝 증가는 항상 수행
    env.step += 1;
}

// 모드 전환 시스템
fn toggle_control_mode(
    keyboard_input: Res<Input<KeyCode>>,
    mut mode: ResMut<ControlMode>,
    mut commands: Commands,
    text_query: Query<Entity, With<OnScreenText>>,
    mut circle_params: ResMut<CircleFlightParams>,
    mut hover_settings: ResMut<HoverSettings>,
) {
    if keyboard_input.just_pressed(KeyCode::Tab) {
        let new_mode = match *mode {
            ControlMode::Keyboard => ControlMode::ReinforcementLearning,
            ControlMode::ReinforcementLearning => ControlMode::CircleTest,
            ControlMode::CircleTest => ControlMode::Keyboard,
        };
        
        // 원형 비행 모드로 전환 시 초기화 재설정
        if new_mode == ControlMode::CircleTest {
            circle_params.initialized = false;
            circle_params.elapsed_time = 0.0;
            
            // 원형 비행용 호버링 설정 조정
            hover_settings.height_pid.kp = 1.5; // 더 강한 P 게인
            hover_settings.height_pid.reset();
        } else if new_mode == ControlMode::Keyboard {
            // 키보드 모드로 전환 시 호버링 설정 기본값으로
            hover_settings.height_pid.kp = 1.0;
            hover_settings.height_pid.reset();
        }
        
        *mode = new_mode;
        
        // 기존 모드 텍스트 제거
        for entity in text_query.iter() {
            commands.entity(entity).despawn();
        }
        
        // 새로운 모드 텍스트 생성
        spawn_mode_text(&mut commands, *mode);
        
        println!("모드 전환: {:?}", new_mode);
    }
}

// 화면에 표시할 텍스트 컴포넌트
#[derive(Component)]
struct OnScreenText;

// 모드 텍스트 생성 함수
fn spawn_mode_text(commands: &mut Commands, mode: ControlMode) {
    commands.spawn((
        TextBundle::from_section(
            format!("Mode: {:?}", mode),
            TextStyle {
                font_size: 30.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        OnScreenText,
    ));
}

// 키보드 입력 테스트 시스템
fn test_keyboard_input(
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
) {
    if (time.elapsed_seconds() % 2.0) < time.delta_seconds() {
        if keyboard_input.pressed(KeyCode::Space) {
            println!("스페이스바가 눌렸습니다!");
        }
        if keyboard_input.pressed(KeyCode::ShiftLeft) {
            println!("Shift가 눌렸습니다!");
        }
        if keyboard_input.pressed(KeyCode::Up) {
            println!("위 화살표가 눌렸습니다!");
        }
        if keyboard_input.pressed(KeyCode::R) {
            println!("R 키가 눌렸습니다!");
        }
    }
}

// 키보드 리셋 시스템
fn keyboard_reset_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut commands: Commands,
    query: Query<Entity, With<DroneAgent>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target_query: Query<Entity, With<Target>>,
    mode: Res<ControlMode>,
    mut hover_settings: ResMut<HoverSettings>,
) {
    if keyboard_input.just_pressed(KeyCode::R) {
        println!("드론 리셋!");
        
        for entity in query.iter() {
            commands.entity(entity).despawn_recursive();
        }
        
        for entity in target_query.iter() {
            commands.entity(entity).despawn_recursive();
        }
        
        // 기본 호버링 높이로 리셋
        hover_settings.hover_height = 5.0;
        hover_settings.height_pid.reset();
        
        let drone_entity = spawn_keyboard_controlled_drone(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, 2.0, 0.0));
        
        // 강화학습 모드일 때 AgentMemory 컴포넌트 추가
        if matches!(*mode, ControlMode::ReinforcementLearning) {
            commands.entity(drone_entity).insert(AgentMemory::default());
            println!("리셋된 드론에 AgentMemory 컴포넌트가 추가되었습니다.");
        }
        
        spawn_target(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, 5.0, 0.0));
    }
}

// UI 업데이트 시스템 (egui 사용)
fn update_learning_stats_ui(
    mut contexts: bevy_egui::EguiContexts,
    agent_query: Query<&AgentMemory>,
    rl_thread: Res<RLThread>,
    mode: Res<ControlMode>,
) {
    use bevy_egui::egui;
    
    if !matches!(*mode, ControlMode::ReinforcementLearning) {
        return;
    }
    
    let (training_steps, best_reward, avg_reward) = rl_patch::get_training_stats(&rl_thread);
    
    // AgentMemory 데이터 가져오기
    let memory = if let Some(memory) = agent_query.iter().next() {
        memory
    } else {
        return;
    };
    
    // 학습 상태 창 표시
    egui::Window::new("강화학습 정보").show(contexts.ctx_mut(), |ui| {
        ui.label(format!("에피소드: {}", memory.total_episodes));
        ui.label(format!("현재 보상: {:.2}", memory.cumulative_reward));
        ui.label(format!("최고 보상: {:.2}", best_reward));
        ui.label(format!("평균 보상: {:.2}", avg_reward));
        ui.label(format!("학습 단계: {}", training_steps));
        ui.label(format!("총 스텝: {}", memory.total_steps));
        
        // 학습 상태 제어
        let is_training = {
            if let Ok(data) = rl_thread.shared_data.lock() {
                data.is_training
            } else {
                false
            }
        };
        
        if ui.button(if is_training { "학습 중지" } else { "학습 시작" }).clicked() {
            if let Ok(mut data) = rl_thread.shared_data.lock() {
                data.is_training = !data.is_training;
            }
        }
        
        if ui.button("모델 저장").clicked() {
            request_save_model(&rl_thread);
        }
    });
}

// 원형 비행 시스템
fn circle_flight_system(
    mut query: Query<(&mut DroneAgent, &Transform)>,
    mut circle_params: ResMut<CircleFlightParams>,
    time: Res<Time>,
    mut hover_settings: ResMut<HoverSettings>,
) {
    // 첫 실행 시 초기화
    if !circle_params.initialized {
        // 드론이 원의 중심 위쪽에서 시작하도록 설정
        for (mut agent, transform) in query.iter_mut() {
            // 시작 위치에서 호버링하도록 높은 추력 설정
            agent.thrust = 0.0; // 중립(호버링) 모드로 변경
            agent.torque = Vec3::ZERO;
            
            // 드론의 현재 위치를 그대로 유지하고, 해당 높이에서 원운동 하도록 설정
            let current_height = transform.translation.y;
            circle_params.height = current_height; // 현재 높이를 유지
            
            // 중심점은 현재 드론 위치로 설정 (회전목마처럼 제자리에서 원운동)
            circle_params.center = transform.translation;
            
            // 호버링 설정 조정
            hover_settings.hover_height = current_height;
            
            println!("회전목마 비행 초기화: 중심=({:.1}, {:.1}, {:.1}), 반경={:.1}, 높이={:.1}", 
                     circle_params.center.x, circle_params.center.y, circle_params.center.z,
                     circle_params.radius, circle_params.height);
        }
        circle_params.initialized = true;
        return; // 초기화 후 첫 프레임은 종료
    }
    
    // 시간 업데이트
    circle_params.elapsed_time += time.delta_seconds();
    let angle = circle_params.elapsed_time * circle_params.speed;
    
    // 원형 경로 계산 (회전목마 형태로 - 현재 높이에서 수평 원운동)
    // 드론 현재 위치를 중심으로 반지름만큼 떨어진 위치를 계산
    let target_x = circle_params.center.x + circle_params.radius * angle.cos();
    let target_z = circle_params.center.z + circle_params.radius * angle.sin();
    let target_y = circle_params.height; // 현재 높이 유지
    
    // 호버링 높이 설정 업데이트
    hover_settings.hover_height = target_y;
    
    let target_position = Vec3::new(target_x, target_y, target_z);
    
    for (mut agent, transform) in query.iter_mut() {
        let current_position = transform.translation;
        
        // 현재 위치에서 목표 위치까지의 방향 벡터
        let direction = target_position - current_position;
        let distance = direction.length();
        
        // 기본 호버링 (수직제어는 control_drone 시스템에서 처리)
        agent.thrust = 0.0;
        
        // 수평면에서의 방향만 고려
        let flat_direction = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
        
        // 전진방향으로 약한 힘을 적용 (이 값은 사용하지 않지만 드론 제어 로직에서 참고 목적으로 남겨둠)
        let _forward_force = flat_direction * 5.0;
        
        // 목표를 향한 회전
        let forward = transform.forward();
        let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        
        // 전진 방향과 목표 방향 사이의 각도 계산
        let dot = flat_forward.dot(flat_direction);
        let cross = flat_forward.cross(flat_direction).y;
        
        // 토크 계산 (좌/우 회전)
        let torque_y = if cross > 0.0 {
            1.0 * (1.0 - dot.max(0.0)) // 왼쪽으로 회전
        } else {
            -1.0 * (1.0 - dot.max(0.0)) // 오른쪽으로 회전
        };
        
        // 수평 안정화를 위한 토크 (주 제어 시스템에서 처리)
        let torque_x = 0.0; // control_drone에서 자세 안정화 담당
        let torque_z = 0.0; // control_drone에서 자세 안정화 담당
        
        // 드론 제어 입력 설정
        agent.torque = Vec3::new(torque_x, torque_y, torque_z);
        
        // 주기적으로 정보 출력 (0.5초마다)
        if (circle_params.elapsed_time * 2.0).round() % 2.0 < time.delta_seconds() {
            println!("회전목마 비행: 각도={:.1}°, 목표=({:.1}, {:.1}, {:.1}), 거리={:.2}", 
                     angle.to_degrees(), 
                     target_x, target_y, target_z, 
                     distance);
        }
    }
}

