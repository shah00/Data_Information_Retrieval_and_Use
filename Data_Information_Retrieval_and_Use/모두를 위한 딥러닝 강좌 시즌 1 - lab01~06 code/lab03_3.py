"""
Lab 3 Minimizing Cost
    lab-03-3-minimizing_cost_tf_optimizer
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# X, Y 정의
X = [1, 2, 3]
Y = [1, 2, 3]

# weight 값을 넣을 tf.Variable 생성. tf.float32 type의 5.0를 초기값으로 가짐.
W = tf.Variable(5.0)

# hypothesis 정의 : H = XW (bias 생략)
hypothesis = X * W

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.1을 주어 cost function을 최소화 하도록 유도. W를 조정해줌.
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # tf.Variable을 이용하기 위한 초기화 작업
    sess.run(tf.global_variables_initializer())
    # 101회 반복
    for step in range(101):
        # train, W 노드를 Session을 이용하여 run. W_val은 갱신된 W 노드의 데이터를 참조
        _, W_val = sess.run([train, W])
        # 회차와 W_val을 출력
        print(step, W_val)

"""
0 5.0
1 1.2666664
2 1.0177778
3 1.0011852
4 1.000079
...
97 1.0
98 1.0
99 1.0
100 1.0
"""