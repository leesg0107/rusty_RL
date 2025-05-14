use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

#[derive(Component)]
pub struct DroneAgent {
    // 에이전트 상태
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    
    // 강화학습 관련 값
    pub reward: f32,
    pub done: bool,
    
    // 드론 특성
    pub thrust: f32,
    pub torque: Vec3,
    
    // 목표 이동 방향 (정규화된 방향 벡터)
    pub target_direction: Vec3,
    // 목표 속도 크기
    pub target_speed: f32,
}

impl Default for DroneAgent {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            reward: 0.0,
            done: false,
            thrust: 0.0,
            torque: Vec3::ZERO,
            target_direction: Vec3::ZERO,
            target_speed: 0.0,
        }
    }
}

// 드론 에이전트 생성 함수 (3D 모델 사용)
pub fn spawn_drone_with_model(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    position: Vec3,
) -> Entity {
    commands.spawn((
        SceneBundle {
            scene: asset_server.load("models/drone.gltf#Scene0"),
            transform: Transform::from_translation(position),
            ..default()
        },
        RigidBody::Dynamic,
        Collider::cuboid(0.5, 0.2, 0.5),
        Damping {
            linear_damping: 0.5,
            angular_damping: 0.5,
        },
        DroneAgent::default(),
    )).id()
}

// 드론의 호버링 상태를 관리하는 리소스 추가
#[derive(Resource)]
pub struct HoverSettings {
    pub base_hover_thrust: f32,  // 기본 호버링에 필요한 추력
    pub hover_height: f32,       // 호버링 목표 높이
    pub height_pid: PIDController, // 높이 유지를 위한 PID 제어기
}

impl Default for HoverSettings {
    fn default() -> Self {
        Self {
            base_hover_thrust: 9.81, // 중력 상쇄를 위한 기본값
            hover_height: 2.0,      // 기본 호버링 높이를 2.0으로 변경
            height_pid: PIDController::new(1.0, 0.1, 0.5, 10.0), // P, I, D, max_output
        }
    }
}

// PID 제어기 구현
pub struct PIDController {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub max_output: f32,
    integral: f32,
    previous_error: f32,
}

impl PIDController {
    pub fn new(kp: f32, ki: f32, kd: f32, max_output: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            max_output,
            integral: 0.0,
            previous_error: 0.0,
        }
    }
    
    pub fn calculate(&mut self, setpoint: f32, current: f32, dt: f32) -> f32 {
        let error = setpoint - current;
        
        // 적분항 (리셋 방지 및 와인드업 방지)
        self.integral = (self.integral + error * dt).clamp(-5.0, 5.0);
        
        // 미분항
        let derivative = if dt > 0.0 { (error - self.previous_error) / dt } else { 0.0 };
        
        // 이전 오차 저장
        self.previous_error = error;
        
        // PID 출력 계산 및 제한
        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.clamp(-self.max_output, self.max_output)
    }
    
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
    }
}

// 드론 제어 시스템 개선
pub fn control_drone(
    mut query: Query<(&mut ExternalForce, &Transform, &mut DroneAgent, &Velocity)>,
    time: Res<Time>,
    gains: Res<ControlGains>,
    mut hover_settings: ResMut<HoverSettings>, // Res에서 ResMut로 변경
) {
    for (mut ext_force, transform, mut agent, velocity) in query.iter_mut() {
        // 현재 상태 업데이트
        agent.position = transform.translation;
        agent.velocity = velocity.linvel;
        agent.rotation = transform.rotation;
        
        // 수직 제어 (PID 사용)
        let dt = time.delta_seconds();
        let current_height = transform.translation.y;
        
        // PID 계산을 위한 값 미리 복사 (가변/불변 참조 충돌 방지)
        let hover_height = hover_settings.hover_height;
        let height_pid_output = {
            let pid = &mut hover_settings.height_pid;
            pid.calculate(hover_height, current_height, dt)
        };
        
        // 상승/하강 기본값 수정 (중립 상태에서는 호버링만 유지)
        let base_thrust = if agent.thrust.abs() < 0.1 {
            // 중립 상태에서는 PID로 고도 유지
            hover_settings.base_hover_thrust + height_pid_output
        } else if agent.thrust > 0.0 {
            // 상승 명령일 때
            hover_settings.base_hover_thrust + agent.thrust * gains.thrust_gain
        } else {
            // 하강 명령일 때 (필요한 만큼만 추력 감소)
            (hover_settings.base_hover_thrust * 0.7 + agent.thrust * gains.thrust_gain).max(0.0)
        };
        
        // 드론의 현재 자세 계산
        let up_direction = transform.rotation * Vec3::Y;
        
        // 이동 방향을 기반으로 목표 기울기 계산 (실제 드론 동작 방식)
        let mut tilt_target = Vec3::ZERO;
        let horizontal_direction = agent.target_direction;
        
        if horizontal_direction.length_squared() > 0.01 && agent.target_speed > 0.1 {
            // 이동 방향이 있을 때, 그 방향으로 기울이기 위한 목표값 계산
            // 목표 속도가 클수록 더 많이 기울임
            let tilt_amount = (agent.target_speed * 0.15).min(0.3); // 최대 ~17도 제한
            
            // 전방 벡터와 오른쪽 벡터가 필요
            let forward = transform.forward();
            let right = transform.right();
            
            // 수평 평면에 투영
            let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            let right_flat = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();
            
            // 우선 전/후방 성분 계산 (피치 - x축 회전)
            let forward_component = horizontal_direction.dot(forward_flat);
            
            // 좌/우 성분 계산 (롤 - z축 회전)
            let right_component = horizontal_direction.dot(right_flat);
            
            // 기울임 방향 계산 - 앞으로 가려면 앞으로 기울임(피치 다운), 뒤로 가려면 뒤로 기울임(피치 업)
            tilt_target.x = -forward_component * tilt_amount;
            
            // 좌/우 기울임 - 오른쪽으로 가려면 오른쪽으로 기울임(롤 우), 왼쪽으로 가려면 왼쪽으로 기울임(롤 좌)
            tilt_target.z = -right_component * tilt_amount;
        }
        
        // 현재 기울기 각도 계산
        let current_euler = transform.rotation.to_euler(EulerRot::XYZ);
        let current_tilt = Vec3::new(current_euler.0, 0.0, current_euler.2);
        
        // 기울기 차이에 따른 토크 계산 (PD 제어)
        let tilt_error = tilt_target - current_tilt;
        let tilt_torque = tilt_error * 10.0 - velocity.angvel * 1.0; // P항 + D항
        
        // Y축(요) 회전은 그대로 적용
        let yaw_torque = Vec3::new(0.0, agent.torque.y * gains.torque_gain, 0.0);
        
        // 추력 적용 (기본 추력)
        let thrust_force = up_direction * base_thrust;
        
        // 토크 적용 (회전력) - 기울임 토크와 요 토크 결합
        let final_torque = Vec3::new(tilt_torque.x, yaw_torque.y, tilt_torque.z);
        
        ext_force.force = thrust_force;
        ext_force.torque = final_torque + agent.torque * Vec3::new(gains.torque_gain, 0.0, gains.torque_gain);
        
        // 감쇠력 조정 - 방향별 차등 감쇠 적용
        let horizontal_vel = Vec3::new(velocity.linvel.x, 0.0, velocity.linvel.z);
        let vertical_vel = Vec3::new(0.0, velocity.linvel.y, 0.0);
        
        // 수평 감쇠 (더 강하게) - x, z 방향
        // 목표 방향으로 이동 시 감쇠를 줄임 (목표 방향으로는 가속 가능하도록)
        let horizontal_damping_factor = if horizontal_direction.length_squared() > 0.01 {
            // 목표 방향과 현재 속도의 일치도 계산
            let vel_dir = horizontal_vel.normalize_or_zero();
            let alignment = horizontal_direction.dot(vel_dir).max(0.0); // 같은 방향일 때만 양수
            
            // 일치도가 높을수록 낮은 감쇠 적용 (0.3 ~ 0.8 범위)
            0.8 - alignment * 0.5
        } else {
            // 기본 감쇠
            0.8
        };
        
        let horizontal_damping = -horizontal_vel * horizontal_damping_factor;
        
        // 수직 감쇠 (방향별 차등)
        let vertical_damping = if velocity.linvel.y > 0.5 {
            // 빠른 상승 속도 제한
            -vertical_vel * 0.8
        } else if velocity.linvel.y < -0.5 {
            // 빠른 하강 속도 제한 (더 강하게)
            -vertical_vel * 1.8  // 1.5에서 1.8로 증가 - 하강 제어 더 강화
        } else {
            // 작은 수직 속도는 약한 감쇠
            -vertical_vel * 0.5
        };
        
        ext_force.force += horizontal_damping + vertical_damping;
        
        // 바닥 근처에서 추가 부양력 (바닥으로 사라지는 문제 방지)
        if transform.translation.y < 0.5 {
            let emergency_lift = (0.5 - transform.translation.y) * 20.0; // 낮을수록 더 강한 부양력
            ext_force.force += Vec3::new(0.0, emergency_lift, 0.0);
            
            // 바닥에 너무 가까우면 위치 강제 보정
            if transform.translation.y < 0.1 {
                // 로그 출력
                println!("바닥 충돌 방지: 드론 위치 강제 보정");
            }
        }
        
        // 기본 자세 안정화 (수평 유지)는 이동이 없을 때만 강하게 적용
        if horizontal_direction.length_squared() < 0.01 || agent.target_speed < 0.1 {
            // 자세 안정화 개선: 기울어짐에 비례한 교정 토크
            let current_up = transform.rotation * Vec3::Y;
            let ideal_up = Vec3::Y;
            
            // 현재 회전 각도 계산 (라디안)
            let tilt_angle = current_up.angle_between(ideal_up);
            let tilt_axis = current_up.cross(ideal_up).normalize_or_zero();
            
            // 각속도에 비례한 댐핑 토크 추가
            let angular_damping_torque = -velocity.angvel * 3.0;
            
            // 기울기에 비례한 복원 토크 계산 (각도가 클수록 더 강한 복원력)
            let tilt_correction = if tilt_angle > 0.01 {
                let correction_strength = 12.0 * tilt_angle; // 9.0에서 12.0으로 증가 - 더 강한 복원력
                tilt_axis * correction_strength
            } else {
                Vec3::ZERO
            };
            
            // 추가적인 수평 복원력 (바닥이 가까울 때 더 강력한 안정화)
            let ground_distance = current_height; // 바닥으로부터의 거리
            let extra_stability = if ground_distance < 1.0 {
                // 1미터 이하에서는 강한 추가 안정화
                tilt_axis * (tilt_angle * 8.0 * (1.0 - ground_distance.max(0.2))) // 5.0에서 8.0으로 증가
            } else {
                Vec3::ZERO
            };
            
            ext_force.torque += tilt_correction + angular_damping_torque + extra_stability;
        }
        
        // 디버그 정보 (필요한 경우)
        if (time.elapsed_seconds() % 5.0) < time.delta_seconds() {
            println!("제어 상태: 높이: {:.2}, PID 출력: {:.2}, 추력: {:.2}, 토크: {:?}", 
                     current_height, height_pid_output, base_thrust, ext_force.torque);
            
            if horizontal_direction.length_squared() > 0.01 {
                println!("이동 상태: 방향: {:?}, 속도: {:.2}, 기울기 목표: {:?}", 
                         horizontal_direction, agent.target_speed, tilt_target);
            }
        }
    }
}

// 키보드 입력을 위한 컴포넌트 추가
#[derive(Component)]
pub struct KeyboardControlled;

// 키보드 제어 시스템 개선
pub fn keyboard_control_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut query: Query<(&mut DroneAgent, &Transform), With<KeyboardControlled>>,
    time: Res<Time>,
    mut hover_settings: ResMut<HoverSettings>,
) {
    // 키보드 제어 값 로깅 (1초마다)
    if (time.elapsed_seconds() % 1.0) < time.delta_seconds() {
        let key_map = [
            (KeyCode::Space, "상승"), 
            (KeyCode::ShiftLeft, "하강"),
            (KeyCode::Up, "전진"), 
            (KeyCode::Down, "후진"),
            (KeyCode::Left, "좌측"), 
            (KeyCode::Right, "우측"),
            (KeyCode::A, "좌회전"), 
            (KeyCode::D, "우회전")
        ];
        
        let pressed_keys: Vec<&str> = key_map.iter()
            .filter(|(key, _)| keyboard_input.pressed(*key))
            .map(|(_, name)| *name)
            .collect();
        
        if !pressed_keys.is_empty() {
            println!("키 입력: {}", pressed_keys.join(", "));
        }
    }
    
    for (mut agent, transform) in query.iter_mut() {
        // 기본값으로 초기화
        let mut thrust = 0.0; // 중립 상태는 0 (호버링)
        let mut torque = Vec3::ZERO;
        let mut movement_direction = Vec3::ZERO;
        let mut movement_speed = 0.0;
        
        // 상승/하강 (스페이스/좌측 Shift)
        if keyboard_input.pressed(KeyCode::Space) {
            thrust = 5.0; // 상승 추력
            // 목표 높이 증가 (호버링 PID 사용 시)
            hover_settings.hover_height += 0.05; // 0.1에서 0.05로 감소 - 더 부드러운 고도 변화
        } else if keyboard_input.pressed(KeyCode::ShiftLeft) {
            thrust = -3.0; // 하강 추력 감소 (-4.0에서 -3.0으로 변경)
            // 목표 높이 감소 (호버링 PID 사용 시)
            hover_settings.hover_height -= 0.05; // 0.1에서 0.05로 감소
            // 최소 높이 제한 (바닥 충돌 방지)
            hover_settings.hover_height = hover_settings.hover_height.max(0.8); // 0.5에서 0.8로 증가
        }
        
        // 전진/후진 (위/아래 화살표) - 완전히 새로운 방식으로 구현
        let mut has_movement = false;
        
        if keyboard_input.pressed(KeyCode::Up) {
            // 전진 방향 설정 (드론의 전방 방향)
            let forward = transform.forward();
            // 수평 평면으로 투영 (수직 방향 제거)
            let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            movement_direction += forward_flat;
            has_movement = true;
        } 
        
        if keyboard_input.pressed(KeyCode::Down) {
            // 후진 방향 설정 (드론의 후방 방향)
            let backward = -transform.forward();
            // 수평 평면으로 투영
            let backward_flat = Vec3::new(backward.x, 0.0, backward.z).normalize_or_zero();
            movement_direction += backward_flat;
            has_movement = true;
        }
        
        // 좌/우 이동 (좌/우 화살표)
        if keyboard_input.pressed(KeyCode::Left) {
            // 좌측 방향 설정 (드론의 좌측 방향)
            let left = -transform.right();
            // 수평 평면으로 투영
            let left_flat = Vec3::new(left.x, 0.0, left.z).normalize_or_zero();
            movement_direction += left_flat;
            has_movement = true;
        } 
        
        if keyboard_input.pressed(KeyCode::Right) {
            // 우측 방향 설정 (드론의 우측 방향)
            let right = transform.right();
            // 수평 평면으로 투영
            let right_flat = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();
            movement_direction += right_flat;
            has_movement = true;
        }
        
        // 이동 방향이 있을 경우 정규화
        if has_movement {
            movement_direction = movement_direction.normalize_or_zero();
            movement_speed = 5.0; // 목표 속도 설정
            
            // 이동 중일 때는 약간의 추가 추력 제공 (고도 유지 향상)
            thrust += 0.5;
        }
        
        // 회전 (A/D 키)
        if keyboard_input.pressed(KeyCode::A) {
            torque.y = 2.0; // 왼쪽으로 회전
        } else if keyboard_input.pressed(KeyCode::D) {
            torque.y = -2.0; // 오른쪽으로 회전
        }
        
        // 값 적용
        agent.thrust = thrust;
        agent.torque = torque;
        agent.target_direction = movement_direction;
        agent.target_speed = movement_speed;
        
        // 키 입력이 있을 때만 디버그 출력
        if thrust != 0.0 || torque.length_squared() > 0.01 || movement_direction.length_squared() > 0.01 {
            println!("키보드 제어: 추력={:.2}, 토크={:?}, 이동방향={:?}, 속도={:.1}", 
                     thrust, torque, movement_direction, movement_speed);
            println!("호버링 목표 높이: {:.2}", hover_settings.hover_height);
        }
    }
}

// 키보드 제어 가능한 드론 생성
pub fn spawn_keyboard_controlled_drone(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position: Vec3,
) -> Entity {
    let drone_entity = commands.spawn((
        TransformBundle::from(Transform::from_translation(position)),
        VisibilityBundle::default(),
        Name::new("Keyboard_Drone"),
        RigidBody::Dynamic,
        ExternalForce::default(),
        Velocity::default(),
        Damping {
            linear_damping: 0.8,  // 감쇠 유지
            angular_damping: 0.8, // 감쇠 유지
        },
        DroneAgent::default(),
        AdditionalMassProperties::Mass(1.0), // 질량 유지
        KeyboardControlled,
        Ccd::enabled(), // 연속 충돌 감지 추가 (바닥 통과 방지)
    )).id();
    
    // 드론 본체 생성
    let body = commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(0.35, 0.1, 0.35))),
            material: materials.add(Color::rgb(0.3, 0.3, 0.7).into()),
            transform: Transform::IDENTITY,
            ..default()
        },
        Name::new("Drone_Body"),
    )).id();
    
    commands.entity(drone_entity).push_children(&[body]);
    
    // 프로펠러 추가
    let propeller_positions = [
        Vec3::new(0.1750, 0.0, 0.0),  // prop1
        Vec3::new(0.0, 0.0, 0.1750),  // prop2
        Vec3::new(-0.1750, 0.0, 0.0), // prop3
        Vec3::new(0.0, 0.0, -0.1750), // prop4
    ];
    
    for (i, pos) in propeller_positions.iter().enumerate() {
        let propeller = commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cylinder {
                    radius: 0.05,
                    height: 0.01,
                    resolution: 20,
                    segments: 1,
                })),
                material: materials.add(Color::rgb(0.8, 0.2, 0.2).into()),
                transform: Transform::from_translation(*pos),
                ..default()
            },
            Name::new(format!("Propeller_{}", i + 1)),
        )).id();
        
        commands.entity(drone_entity).push_children(&[propeller]);
    }
    
    // 콜라이더 추가 - 크기 약간 증가하여 바닥 충돌 개선
    commands.entity(drone_entity).insert(Collider::cylinder(0.1, 0.35));
    
    println!("새 드론 생성: 위치 = {:?}", position);
    
    drone_entity
}

// 제어 게인 리소스
#[derive(Resource)]
pub struct ControlGains {
    pub thrust_gain: f32,
    pub torque_gain: f32,
}

impl Default for ControlGains {
    fn default() -> Self {
        Self {
            thrust_gain: 10.0,
            torque_gain: 5.0,
        }
    }
} 