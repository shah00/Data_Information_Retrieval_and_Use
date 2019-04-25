"""
Lab 3 Minimizing Cost
    lab-03-2-minimizing_cost_gradient_update
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x_data, y_data값 정의
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = (1)인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
W = tf.Variable(tf.random_normal([1]), name="weight")

# X가 tf.float32 type을 데이터로 갖는 placeholder 노드를 참조(dictionary를 feed 해주어야 함)
# Y가 tf.float32 type을 데이터로 갖는 placeholder 노드를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis 정의 : H = X * W (bias 값 생략)
hypothesis = X * W

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient descent를 적용하여 cost function을 minimize.
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # sess.run()으로 위의 Variable들을 초기화
    sess.run(tf.global_variables_initializer())
    # 21회 반복
    for step in range(21):
        # {X: x_data, Y: y_data}를 feed한 cost, W 노드를 Session을 통해 run하고, 결과 값(노드의 바뀐 데이터)를
        # cost_val, W_val가 참조
        _, cost_val, W_val = sess.run(
            [update, cost, W], feed_dict={X: x_data, Y: y_data}
        )
        # step, cost_val, W_val 값 출력
        print(step, cost_val, W_val)

"""
0 1.93919 [ 1.64462376]
1 0.551591 [ 1.34379935]
2 0.156897 [ 1.18335962]
3 0.0446285 [ 1.09779179]
4 0.0126943 [ 1.05215561]
5 0.00361082 [ 1.0278163]
6 0.00102708 [ 1.01483536]
7 0.000292144 [ 1.00791216]
8 8.30968e-05 [ 1.00421977]
9 2.36361e-05 [ 1.00225055]
10 6.72385e-06 [ 1.00120032]
11 1.91239e-06 [ 1.00064015]
12 5.43968e-07 [ 1.00034142]
13 1.54591e-07 [ 1.00018203]
14 4.39416e-08 [ 1.00009704]
15 1.24913e-08 [ 1.00005174]
16 3.5322e-09 [ 1.00002754]
17 9.99824e-10 [ 1.00001466]
18 2.88878e-10 [ 1.00000787]
19 8.02487e-11 [ 1.00000417]
20 2.34053e-11 [ 1.00000226]
"""