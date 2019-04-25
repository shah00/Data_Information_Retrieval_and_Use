"""
# Lab 3 Minimizing Cost
    lab-03-1-minimizing_cost_show_graph
"""

# tensorflow를 tf라는 이름으로 import
# matplotlib package 내 pyplot module을 plt라는 이름으로 import
import tensorflow as tf
import matplotlib.pyplot as plt

# X, Y값 정의
X = [1, 2, 3]
Y = [1, 2, 3]

# W가 tf.float32 type을 데이터로 갖는 placeholder 노드를 참조(dictionary를 feed 해주어야 함)
W = tf.placeholder(tf.float32)

# hypothesis 정의 : H = X * W (bias 값 생략)
hypothesis = X * W

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost function을 시각화하기 위한 두 개의 list
W_history = []
cost_history = []

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # 80회 반복
    for i in range(-30, 50):
        # curr_W의 값은 i * 0.1
        curr_W = i * 0.1
        # curr_cost가 {W: curr_W}를 feed하여 Session을 통해 run한 cost 노드의 데이터를 참조
        curr_cost = sess.run(cost, feed_dict={W: curr_W})
        # W_history list에 curr_W 값을 append
        W_history.append(curr_W)
        # cost_history list에 curr_cost 값을 append
        cost_history.append(curr_cost)

# W_history를 x축으로, cost_history를 y축으로 하는 그래프를 작성, 출력
plt.plot(W_history, cost_history)
plt.show()
