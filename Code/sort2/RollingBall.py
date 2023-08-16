from vpython import *

scene = canvas(width=800, height=600, center=vector(0, 0, 0), background=color.white)
floor = box(pos=vector(0, -0.5, 0), size=vector(10, 0.1, 1), color=color.gray(0.7))

ball = sphere(pos=vector(-4.5, 0, 0), radius=0.5, color=color.blue)

# 속도와 가속도 정의
velocity = vector(0.03, 0, 0)
acceleration = vector(0.001, 0, 0)

while True:
    rate(100)  # 루프 속도 제어

    # 속도와 가속도 적용
    velocity += acceleration

    # 좌우로 반복하는 움직임 구현
    if ball.pos.x > 4.5 or ball.pos.x < -4.5:
        velocity.x *= -1

    ball.pos += velocity
