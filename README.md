# RaccoonBot OpenVLA 텀프로젝트

피지컬AI 텀프로젝트로 진행한 RaccoonBot + OpenVLA 실험 정리입니다. 기본 제공 코드에서 MuJoCo 데이터셋을 확장하고, RLDS/TFDS 변환과 짧은 LoRA 테스트를 진행한 뒤, OpenVLA action을 RaccoonBot에서 조금 더 안정적으로 실행할 수 있도록 client 쪽 action mapping을 수정했습니다.

## 1. 프로젝트에서 한 일

전체적으로 한 일은 다음과 같습니다.

- MuJoCo demonstration dataset 확장
- RLDS / TFDS dataset rebuild
- 100 step short LoRA test
- OpenVLA 7D action을 RaccoonBot 4DOF 구조에 맞게 실행하는 client code 개선
- MuJoCo에서 before / after 결과 비교
- 실제 RaccoonBot에서 fixed-layout grasp-and-lift 실험

## 2. Dataset Extension

과제 안내에서 제시한 extension 항목 중에서 저는 다음 방향을 선택했습니다.

- 새로운 task 추가: `push`, `lift`
- 더 다양한 language instruction 추가
- 새로운 object type 추가: `cube`, `sphere`

### 2.1 Grasp / Push Dataset

먼저 기존 cylinder 환경에서 grasp 외에 push task metadata와 다양한 instruction template을 추가했습니다.

예시 instruction은 다음과 같습니다.

```text
grasp the red cylinder
grab the green cylinder
pick up the blue cylinder
lift the yellow cylinder
push the red cylinder
move the green cylinder forward
```

생성한 extended dataset 요약은 다음과 같습니다.

- 총 episode 수: 20
- train / validation split: 18 / 2
- task count: grasp 7개, push 13개
- color count: red / blue / green / yellow 각각 5개
- unique instruction 수: 14개

관련 파일은 아래에 있습니다.

```text
Mujoco/raccoon_grasp_push_extended_dataset.py
project_evidence/summaries/extended_dataset_summary.json
project_evidence/logs/rlds_convert_extended.log
project_evidence/logs/tfds_build_extended.log
```

### 2.2 Cube / Sphere 추가

추가로 기존 colored cylinder XML을 바탕으로 `Raccoon_multishape_objects.xml`을 만들었습니다. 여기서는 blue object를 cube로, green object를 sphere로 바꾸었습니다.

```text
red: cylinder
blue: cube
green: sphere
yellow: cylinder
```

관련 파일은 아래에 있습니다.

```text
client_improvements/create_multishape_xml.py
client_improvements/Raccoon_multishape_objects.xml
client_improvements/evidence/multishape_scene_summary.json
```

## 3. RLDS / TFDS Rebuild와 LoRA Test

확장한 MuJoCo episode를 intermediate format으로 변환한 뒤 TFDS dataset을 다시 build했습니다. TFDS builder에서는 `INTERMEDIATE_ROOT`를 확장 dataset 경로로 바꾸어 사용했습니다.

수정 파일:

```text
Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py
```

TFDS build 결과 train 18개, val 2개가 생성되었습니다. 이후 100 step short LoRA test도 진행했습니다. full training을 끝까지 돌린 것은 아니고, 수정한 dataset이 OpenVLA fine-tuning script에서 정상적으로 로드되고 checkpoint 저장까지 되는지 확인하는 목적이었습니다.

관련 로그:

```text
project_evidence/logs/lora_100step_retry.log
project_evidence/logs/openvla_server_lora_checkpoint.log
project_evidence/logs/hf_checkpoint_download.log
project_evidence/logs/openvla_server_cuda_attempt.log
```

## 4. Code Improvement

가장 많이 수정한 부분은 OpenVLA server가 반환하는 action을 실제 RaccoonBot 동작으로 바꾸는 client 쪽 실행부입니다.

OpenVLA는 기본적으로 아래와 같은 7D action을 출력합니다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

하지만 RaccoonBot은 실제로 xyz 이동과 gripper 중심으로 동작하는 4DOF 구조라서, baseline에서는 target 근처로 가더라도 gripper가 닫히지 않거나, lift까지 이어지지 않는 경우가 많았습니다.

그래서 아래 파일들을 추가했습니다.

```text
client_improvements/action_postprocess.py
client_improvements/action_target_assist_client.py
client_improvements/action_target_assist_client_v2.py
client_improvements/summarize_task_evidence.py
```

### 4.1 7D-to-4DOF Action Mapping

`action_target_assist_client.py`와 `action_target_assist_client_v2.py`에서는 task를 단계별로 나누어 실행하도록 했습니다.

lift/grasp 계열은 다음 단계로 실행됩니다.

```text
approach -> descend -> close -> lift
```

push task는 다음 단계로 실행됩니다.

```text
approach -> descend -> push
```

이 방식은 OpenVLA inference request를 유지하면서도, RaccoonBot의 실제 동작 구조에 맞게 xyz 이동과 gripper command를 보정하는 방식입니다.

### 4.2 Speed / Timing Improvement

`action_target_assist_client_v2.py`에서는 매 step마다 서버에 inference를 요청하지 않고, `--request_every_n_steps` 옵션을 두었습니다. 실험에서는 주로 `request_every_n_steps=3`을 사용했습니다.

이 경우 모든 step에서 서버 요청을 보내는 방식보다 request 수가 약 1/3 수준으로 줄어들었습니다. 예를 들어 blue cube lift 실험에서는 다음과 같은 결과가 나왔습니다.

- every-step request: 28 steps, 28 requests, 평균 step time 약 811.7 ms
- every-3-step request: 28 steps, 10 requests, 평균 step time 약 581.2 ms

즉, 같은 lift 성공을 유지하면서도 inference request 수와 step time을 줄일 수 있었습니다.

### 4.3 Timing / Action Log

각 실험에서는 아래 항목을 csv로 기록했습니다.

- raw OpenVLA action
- assisted action
- inference latency
- motion execution time
- total step time
- gripper command
- object lift distance
- object push distance
- server request count

이를 통해 실패 원인과 개선 효과를 더 명확하게 확인할 수 있었습니다.

## 5. Before / After 결과

### 5.1 Baseline 실패

baseline client로 red / blue cylinder를 실행했을 때는 안정적인 grasp가 잘 되지 않았습니다. action log를 확인해보면 gripper command가 계속 0에 가까운 경우가 많아서, target 근처로 가더라도 실제로 물체를 잡지 못하는 문제가 있었습니다.

관련 evidence:

```text
client_improvements/evidence/baseline_red.log
client_improvements/evidence/baseline_blue.log
client_improvements/evidence/baseline_failed_red.png
client_improvements/evidence/baseline_failed_blue.png
```

### 5.2 MuJoCo 개선 결과

target-assisted action mapping을 적용한 뒤에는 MuJoCo에서 red / blue cylinder grasp-and-lift에 성공했습니다.

```text
client_improvements/evidence/target_assist_red_success.csv
client_improvements/evidence/target_assist_blue_success.csv
client_improvements/evidence/target_assist_red_success.mp4
client_improvements/evidence/target_assist_blue_success.mp4
```

측정된 lift 결과:

- red cylinder: 약 0.0162 m
- blue cylinder: 약 0.0159 m

### 5.3 New Object / New Task 결과

추가로 cube와 sphere를 넣은 multishape scene에서 lift / push task를 실험했습니다.

주요 결과는 다음과 같습니다.

- blue cube lift: 성공, lift 약 0.0120 m
- green sphere push: 성공, 이동 약 0.0104 m
- red cylinder push: 약 0.0090 m 이동
- green sphere lift: lift 중 떨어져서 성공으로 보지는 않음

관련 evidence:

```text
client_improvements/evidence/v2_task_evidence_summary.json
client_improvements/evidence/v2_final_evidence_summary.json
client_improvements/evidence/v2_blue_cube_lift.csv
client_improvements/evidence/v2_green_sphere_push.csv
client_improvements/evidence/v2_red_cylinder_push_tuned.csv
client_improvements/evidence/v2_green_sphere_lift.csv
```

## 6. 실제 RaccoonBot 실험

실제 RaccoonBot에서도 fixed-layout 방식으로 red cylinder grasp-and-lift를 실행했습니다. target cylinder는 로봇 기준 대략 x = 0 cm, y = 17 cm 위치에 두고 실험했습니다.

카메라 없이 안전성과 반복성을 위해 target pose를 고정한 상태에서, OpenVLA inference pipeline과 개선한 action mapping을 이용해 실행했습니다.

결과적으로 실제 로봇 로그에서 `close`, `lift` stage까지 진행되었고, `gripper_cmd=1.0`으로 gripper가 닫혔습니다. 마지막 lift 값은 약 0.0122 m로 기록되었습니다.

관련 evidence:

```text
client_improvements/evidence/real_target_assist_red_success.csv
client_improvements/evidence/real_target_assist_red_success.log
client_improvements/evidence/real_robot_red_grasp_lift.mp4
```

## 7. 실행 방법

### 7.1 OpenVLA server 연결

A100 서버에서 OpenVLA server를 실행하고, 로컬 PC에서는 SSH tunnel을 사용했습니다.

```bash
ssh -L 8000:127.0.0.1:8000 root@qlak315.iptime.org -p 11170
```

### 7.2 MuJoCo lift 실행 예시

```bash
python action_target_assist_client_v2.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_multishape_objects.xml --target_color blue --object_shape cube --task lift --instruction "lift the blue cube" --use_viewer --max_steps 60 --target_x 0.0 --target_y 0.17 --max_delta_xyz 0.008 --speed 55 --settle_seconds_per_action 0.22 --grasp_z 0.022 --request_every_n_steps 3 --output_dir evidence/v2_rollouts --episode_id 1 --action_log_csv evidence/v2_blue_cube_lift.csv
```

### 7.3 MuJoCo push 실행 예시

```bash
python action_target_assist_client_v2.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_multishape_objects.xml --target_color green --object_shape sphere --task push --instruction "move the green sphere forward" --use_viewer --max_steps 70 --target_x 0.0 --target_y 0.17 --max_delta_xyz 0.008 --speed 55 --settle_seconds_per_action 0.22 --push_z 0.026 --push_distance 0.045 --request_every_n_steps 3 --output_dir evidence/v2_rollouts --episode_id 4 --action_log_csv evidence/v2_green_sphere_push.csv
```

### 7.4 실제 RaccoonBot 실행 예시

```bash
python action_target_assist_client.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer --use_real_robot --max_steps 60 --target_x 0.0 --target_y 0.17 --max_delta_xyz 0.006 --speed 25 --settle_seconds_per_action 0.4 --real_settle_seconds 1.0 --real_go_home_on_exit --grasp_z 0.022 --output_dir evidence/real_rollouts --episode_id 2 --action_log_csv evidence/real_target_assist_red_success.csv
```

## 8. 한계점

- 실제 로봇 실험은 fixed-layout 방식
- LoRA는 100 step 짧은 테스트만 수행
- sphere lift는 실패했습니다. 대신 sphere push는 성공 evidence로 사용
- push task는 이동 거리가 크지는 않았지만 green sphere push에서 1 cm 이상 이동

## 9. MuJoCo Dataset Episode Visualization

확장 dataset 자체의 episode도 프레임을 연결하여 확인했습니다. 실제 로봇 영상과 별도로, MuJoCo에서 생성한 demonstration이 의도한 순서대로 진행되는지 확인하기 위한 자료입니다.

```text
client_improvements/visualize_dataset_episode.py
project_evidence/videos/extended_push_episode_000003.mp4
project_evidence/screenshots/extended_push_episode_000003_contact_sheet.png
```

## 10. Pitch 제어와 Sphere Lift 실패 분석

현재 `raccoon_env.py`에서는 OpenVLA가 출력하는 7D action 중 xyz 이동과 gripper command를 중심으로 실행하고, `droll`, `dpitch`, `dyaw`는 직접 추종하지 않습니다. RaccoonBot은 full 6D end-effector 제어가 가능한 구조가 아니기 때문입니다. 실제 로봇 IK에서는 4번 관절을 다음 관계식으로 계산하여 gripper가 아래 방향을 유지하도록 보정했습니다.

```text
th4 = -(th2 + th3) - 90 deg
```

즉 pitch를 완전히 고려하지 않은 것이 아니라, 임의 pitch 명령을 추종하는 대신 grasp 안정성을 위해 고정 자세로 제한한 것입니다. 가변 pitch 제어를 추가하려면 4번 관절 범위와 충돌 가능성을 함께 검증해야 하므로, 이번 실험에서는 안전한 고정 자세를 사용했습니다.

Sphere lift가 불안정했던 이유도 이 제한과 관련이 있습니다. Cylinder와 cube는 gripper가 양쪽에서 접촉할 수 있는 면이 비교적 안정적이지만, sphere는 접촉점이 좁고 곡면을 따라 미끄러지기 쉽습니다. 여기에 접근 위치 오차와 고정 pitch 자세가 함께 작용하면 gripper 중심이 조금만 어긋나도 lift 과정에서 물체가 빠질 수 있습니다. 실제 viewer에서도 sphere가 들어 올려지는 중 떨어졌고, 로그의 최대 lift 값도 약 0.0022 m로 작았습니다. 따라서 sphere는 push 성공 결과를 사용하고 lift는 한계 사례로 남겼습니다.

## 11. 정리

이번 프로젝트에서는 dataset extension, RLDS/TFDS rebuild, short LoRA test, action mapping 개선, MuJoCo 결과, 실제 RaccoonBot 결과까지 한 번에 연결해보는 것을 목표로 했습니다.

OpenVLA의 7D action을 그대로 실행했을 때 생기는 문제를 로그로 확인하고, RaccoonBot의 4DOF 구조에 맞게 staged action mapping을 추가하여 baseline보다 실행 과정이 더 명확해졌고, MuJoCo에서는 cylinder/cube lift와 sphere push를 확인했으며, 실제 RaccoonBot에서도 fixed-layout grasp-and-lift를 성공시켰습니다.
