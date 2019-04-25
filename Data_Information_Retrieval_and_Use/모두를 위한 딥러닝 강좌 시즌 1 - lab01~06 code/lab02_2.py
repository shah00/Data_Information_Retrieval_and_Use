"""
Lab 2 Linear Regression
    lab-02-2-linear_regression_feed
"""

# tensorflow를 tf란 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = (1)인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = (1)인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# X가 type은 float32, shape : (None)인 placeholder 참조(dictionary를 feed 해주어야 함)
# Y가 type은 float32, shape : (None)인 placeholder 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# hypothesis 정의 : H = X * W + b
hypothesis = X * W + b

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning_rate를 0.01(강의 내에서의 알파 값)으로 두어 Gradient Descent을 적용, cost function을 minimize하도록 설정.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # tf.Variable을 이용하기 위한 초기화 작업
    sess.run(tf.global_variables_initializer())

    # 2001회 반복
    for step in range(2001):
        # 훈련 적용. cost_val, W_val, b_val이 각각 변화되는 cost function, W(weight), b(bias)값을 참조
        # {X: [1, 2, 3], Y: [1, 2, 3]}를 feed하고 그래프(노드 집합)을 Session에 넣어서 run
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}
        )
        # 20회마다 step, cost_val, W_val, b_val 출력
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # {X: }을 feed한 다음, hypothesis(가설)을 Session에 넣어 run. 그 값을 출력
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 0]
    """
    0 3.5240757 [2.2086694] [-0.8204183]
    20 0.19749963 [1.5425726] [-1.0498911]
    ...
    1980 1.3360998e-05 [1.0042454] [-0.00965055]
    2000 1.21343355e-05 [1.0040458] [-0.00919707]
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    """

    # 2001회 반복
    for step in range(2001):
        # 훈련 적용. cost_val, W_val, b_val이 각각 변화되는 cost function, W(weight), b(bias)값을 참조
        # {X: [1, 2, 3], Y: [1, 2, 3]}를 feed하고 그래프(노드 집합)을 Session에 넣어서 run
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b],
            feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]},
        )
        # 20회마다 step, cost_val, W_val, b_val 출력
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # {X: }을 feed한 다음, hypothesis(가설)을 Session에 넣어 run. 그 값을 출력
    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    # Learns best fit W:[ 1.],  b:[ 1.1]
    """
    0 1.2035878 [1.0040361] [-0.00917497]
    20 0.16904518 [1.2656431] [0.13599995]
    ...
    1980 2.9042917e-07 [1.00035] [1.0987366]
    2000 2.5372992e-07 [1.0003271] [1.0988194]
    [6.1004534]
    [3.5996385]
    [2.5993123 4.599964 ]
    """