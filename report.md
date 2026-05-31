# 텀 프로젝트 보고서

< Physical AI Term Project >

과목명 : 피지컬AI  
학  번 : 2024405002  
이  름 : 김민기  

---

# - 목 차 -

1. 프로젝트 개요  
2. Dataset Extension  
3. RLDS / TFDS 변환 및 LoRA Test  
4. Code Improvement  
5. 실험 결과  
6. 한계점 및 결론  

---

# 1. 프로젝트 개요

이번 프로젝트에서는 수업에서 제공된 RaccoonBot OpenVLA 예제를 실행한 뒤, 데이터셋과 client 코드를 직접 수정하여 기존보다 다양한 동작을 수행하도록 하였다. 전체 흐름은 MuJoCo에서 demonstration을 생성하고, 이를 RLDS / TFDS 형식으로 변환한 뒤, OpenVLA server와 client를 연결하여 결과를 확인하는 방식이다.

기존 예제는 colored cylinder를 잡는 동작이 중심이었다. 또한 OpenVLA가 출력하는 7D action을 그대로 사용하면 RaccoonBot의 구조와 맞지 않아 물체 근처까지 이동하더라도 gripper가 적절한 시점에 닫히지 않는 경우가 있었다. 따라서 이번 프로젝트에서는 dataset extension과 action mapping 개선을 중심으로 실험하였다.

# 2. Dataset Extension

## 가. Task와 instruction 확장

기존 grasp 중심의 데이터에 push task를 추가하였다. instruction도 한 문장만 반복하지 않고 다음과 같이 여러 표현을 사용하였다.

```text
grasp the red cylinder
grab the green cylinder
pick up the blue cylinder
lift the yellow cylinder
push the red cylinder
move the green cylinder forward
```

생성한 extended dataset은 총 20개의 episode로 구성하였다. 색상은 red, blue, green, yellow가 각각 5개씩 포함되도록 하였고, task는 grasp 7개와 push 13개로 구성하였다. 서로 다른 instruction은 총 14개이다.

## 나. Object type 확장

추가 실험에서는 cylinder 외에도 cube와 sphere를 사용하였다. 기존 XML을 바탕으로 multishape scene을 만들고, blue object는 cube, green object는 sphere로 변경하였다.

```text
red: cylinder
blue: cube
green: sphere
yellow: cylinder
```

## 다. Episode 시각화

확장한 dataset이 실제로 어떤 동작을 포함하는지 확인하기 위해 MuJoCo episode의 프레임을 연결하여 영상으로 저장하였다. 주요 프레임은 contact sheet로도 정리하였다.

```text
project_evidence/videos/extended_push_episode_000003.mp4
project_evidence/screenshots/extended_push_episode_000003_contact_sheet.png
```

# 3. RLDS / TFDS 변환 및 LoRA Test

생성한 MuJoCo demonstration은 OpenVLA 학습 pipeline에서 사용할 수 있도록 intermediate format으로 변환하고 TFDS dataset으로 다시 build하였다. TFDS build 결과 train split 18개, validation split 2개가 생성되었다.

이후 확장한 dataset이 fine-tuning script에서 정상적으로 읽히는지 확인하기 위해 100 step short LoRA test를 수행하였다. 학습 로그에서 dataset statistics 계산, 100 step 진행, checkpoint 저장까지 완료된 것을 확인하였다. 이번 LoRA test는 성능 향상을 검증하기 위한 full training이 아니라 pipeline 동작 확인을 위한 짧은 테스트이다.

# 4. Code Improvement

## 가. 7D-to-4DOF action mapping

OpenVLA는 다음과 같은 7D action을 출력한다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

하지만 RaccoonBot은 full 6D end-effector 제어가 가능한 로봇이 아니므로, xyz 이동과 gripper command를 중심으로 실행하도록 mapping을 수정하였다. 또한 한 번에 동작을 수행하지 않고 단계별로 나누었다.

```text
lift / grasp: approach -> descend -> close -> lift
push:         approach -> descend -> push
```

이 방식으로 물체 근처에 접근한 뒤 gripper를 닫고 들어 올리는 순서를 명확하게 만들 수 있었다.

## 나. 실행 속도와 log 개선

기존에는 매 step마다 OpenVLA server에 inference 요청을 보냈다. 개선한 client에서는 `request_every_n_steps` 옵션을 추가하여 필요한 간격마다 요청을 보내도록 하였다.

Blue cube lift 실험에서 every-step 방식은 28번의 요청이 필요했고 평균 step time은 약 811.7 ms였다. 반면 3 step마다 요청한 방식은 같은 28 step 동안 요청 수가 10번으로 줄었고 평균 step time도 약 581.2 ms로 감소하였다. Lift 성공 결과는 유지되었다.

각 step에서는 inference latency, motion execution time, raw action, assisted action, gripper command, lift 거리와 push 거리를 CSV로 기록하였다. 이를 통해 실패 원인과 개선 결과를 수치로 확인할 수 있도록 하였다.

## 다. Pitch 제어 방식

현재 MuJoCo 환경에서는 OpenVLA의 `droll`, `dpitch`, `dyaw`를 직접 추종하지 않는다. 실제 RaccoonBot에서도 임의의 pitch를 자유롭게 제어하는 대신, 역기구학 계산에서 4번 관절을 다음과 같이 계산하여 gripper가 아래 방향을 유지하도록 하였다.

```text
th4 = -(th2 + th3) - 90 deg
```

즉 pitch를 완전히 무시한 것은 아니지만, 이번 프로젝트에서는 충돌 가능성과 관절 범위를 고려하여 고정된 자세를 사용하였다. 가변 pitch 제어는 추가 검증이 필요한 후속 과제로 남겼다.

# 5. 실험 결과

## 가. Baseline과 개선 결과

Baseline client에서는 red / blue cylinder를 안정적으로 잡지 못하였다. End-effector가 물체 근처로 이동하더라도 gripper command가 0에 가까운 경우가 많아 grasp-and-lift로 이어지지 않았다.

Action mapping을 수정한 뒤에는 MuJoCo에서 red / blue cylinder를 잡아 들어 올릴 수 있었다.

```text
red cylinder lift:  약 0.0162 m
blue cylinder lift: 약 0.0159 m
```

## 나. New object와 new task 결과

Multishape scene에서 cube lift와 sphere push를 추가로 확인하였다.

```text
blue cube lift:     성공, lift 약 0.0120 m
green sphere push:  성공, 이동 약 0.0104 m
red cylinder push:  이동 약 0.0090 m
green sphere lift:  실패
```

Sphere lift는 물체가 올라가는 과정에서 떨어졌다. Sphere는 cylinder나 cube와 달리 평평한 접촉면이 없어서 gripper 중심이 조금만 어긋나도 곡면을 따라 미끄러질 수 있다. 또한 현재는 고정 pitch 자세를 사용하므로 물체 형상에 맞게 접촉 방향을 세밀하게 조절하기 어렵다. 실제 로그에서도 sphere의 최대 lift 값은 약 0.0022 m로 작았다. 따라서 sphere lift는 성공 결과로 포함하지 않고 한계 사례로 정리하였다.

## 다. 실제 RaccoonBot 결과

실제 RaccoonBot에서도 red cylinder grasp-and-lift를 수행하였다. 안전성과 반복성을 위해 target cylinder를 로봇 기준 x = 0 cm, y = 17 cm 부근에 둔 fixed-layout 방식으로 실험하였다.

실험 로그에서는 `close`, `lift` stage까지 진행되었고 `gripper_cmd=1.0`으로 gripper가 닫힌 것을 확인하였다. 마지막 lift 값은 약 0.0122 m로 기록되었다.

```text
client_improvements/evidence/real_robot_red_grasp_lift.mp4
```

# 6. 한계점 및 결론

이번 프로젝트에서는 dataset extension, RLDS / TFDS rebuild, short LoRA test, action mapping 개선, MuJoCo episode 시각화, 실제 RaccoonBot 실행까지 전체 pipeline을 한 번 연결하였다.

가장 의미 있었던 부분은 OpenVLA의 action을 그대로 실행하는 데서 끝내지 않고, RaccoonBot의 구조에 맞게 단계별 mapping을 추가한 점이다. 그 결과 baseline보다 grasp-and-lift 과정이 명확해졌고, inference 요청 수를 줄이면서 실행 시간도 단축할 수 있었다.

한계점도 있다. 실제 로봇 실험은 임의 위치의 물체를 카메라로 찾아 집는 방식이 아니라 fixed-layout 방식이다. LoRA 역시 100 step short test만 수행했기 때문에 학습에 따른 성능 향상을 확인한 것은 아니다. 또한 sphere lift 결과를 통해 물체 형상에 따라 gripper 접촉과 pitch 제어가 중요하다는 점을 확인하였다.

추후에는 다양한 물체 형상을 포함한 demonstration을 더 많이 생성하고, 관절 범위 안에서 작은 pitch offset을 적용하여 sphere와 같이 미끄러지기 쉬운 물체의 grasp 안정성을 비교해볼 수 있을 것이다.
