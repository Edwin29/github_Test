import cv2
import numpy as np

def detect_ball(frame):
    # 영상을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (잡음을 감소시킴)
    blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)

    # 허프 원 변환을 위한 원 검출
    circles = cv2.HoughCircles(blurred_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=100, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        # 검출된 원이 있을 경우
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, radius) in circles:
            # 원 주변에 테두리 그리기
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            
            # 테두리 그리기
            cv2.drawContours(frame, [np.array([(x, y)])], -1, (0, 0, 255), 2)

    return frame

# 웹캠이나 비디오 파일을 열기
video_capture = cv2.VideoCapture(0)  # 0은 기본 웹캠, 파일 경로를 지정하면 해당 비디오 파일 사용 가능

while True:
    # 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break

    # 공 감지 함수 호출
    processed_frame = detect_ball(frame)

    # 결과 영상 출력
    cv2.imshow("Ball Detection", processed_frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
video_capture.release()
cv2.destroyAllWindows()
