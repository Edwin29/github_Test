import cv2
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

# 학습된 MobileNetV2 모델 로드
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 분류를 위한 추가적인 레이어 설정
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 학습된 모델 로드 (미리 학습된 가중치를 사용하려면 해당 가중치 파일이 필요합니다.)
model.load_weights('pretrained_model_weights.h5')

# 웹캠이나 비디오 파일을 열기
video_capture = cv2.VideoCapture(0)  # 0은 기본 웹캠, 파일 경로를 지정하면 해당 비디오 파일 사용 가능

while True:
    # 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break

    # 감지된 원들에 대해서 공인지 아닌지 판별
    # 여기서는 단순히 공이 아닌 것으로 간주되는 경우 빨간색 원을 그리고, 공으로 판별되는 경우 녹색 원을 그립니다.
    # 실제로는 더 복잡한 분류 모델을 사용해야 하며, 이 모델을 훈련하는 과정은 더 많은 데이터와 라벨이 필요합니다.
    # 딥러닝 모델을 학습시키는 방법은 별도의 학습 가이드가 필요합니다.
    # 여기서는 예시로 단순히 모든 원을 공으로 판별하는 것으로 가정합니다.
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=100, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, radius) in circles:
            # 원 주변에 테두리 그리기
            if model.predict(np.expand_dims(cv2.resize(frame[y-radius:y+radius, x-radius:x+radius], (224, 224)), axis=0)) > 0.5:
                # 공으로 판별될 경우 녹색 원 그리기
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            else:
                # 공이 아닌 것으로 판별될 경우 빨간색 원 그리기
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # 결과 영상 출력
    cv2.imshow("Ball Detection", frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
video_capture.release()
cv2.destroyAllWindows()
