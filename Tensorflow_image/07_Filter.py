1. 필터셋은 3차원 형태로 된 가중치의 모음
2. 필터셋 하나는 앞선 레이어의 결과인 "특징맵" 전체를 본다.
3. 필터셋 개수 만큼 특징맵을 만든다.

Conv2D(3, kernel_size=5, activation='swish')
Conv2D(6, kernel_size=5, activation='swish')

Convolution 연산의 이해
(8,8,1) (3,3,1) (6,6)