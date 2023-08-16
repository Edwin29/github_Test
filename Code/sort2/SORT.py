import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    cap = cv2.VideoCapture("./RollingBall.mp4")  # 웹캠 사용시 0, 비디오 파일 사용시 경로

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오 읽기 오류")
            break

        # 이미지 처리 (공 감지 및 표시)
        ball_frame = detect_ball(frame)

        # 결과 시각화
        plt.imshow(ball_frame[..., ::-1])  # OpenCV에서 이미지는 BGR로 저장되기 때문에 RGB로 변환해줍니다.
        plt.draw()
        plt.pause(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_ball(frame):
    # 색상 범위 설정 (이 부분은 공의 색상에 따라 적절하게 조정해야 합니다.)
    lower_color = np.array([0, 100, 100])  # HSV 기준, 파란색
    upper_color = np.array([20, 255, 255]) # HSV 기준, 파란색

    # 이미지를 HSV로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 마스크 생성
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # 노이즈 제거를 위한 모폴로지 연산 적용
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 공의 중심 찾기
    ball_frame = frame.copy()
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 최소 반지름 이상인 경우 공으로 간주하고 표시
        if radius > 10:
            cv2.circle(ball_frame, center, radius, (0, 255, 255), 2)
            cv2.circle(ball_frame, center, 5, (0, 0, 255), -1)

    return ball_frame

if __name__ == "__main__":
    main()
