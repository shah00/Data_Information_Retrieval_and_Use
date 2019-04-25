"""
Lab 2 Linear Regression
    lab-02-3-linear_regression_tensorflow
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# x_train 정의. shape : (4)
# y_train 정의. shape : (4)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# weight 값을 넣을 tf.Variable 생성. tf.float32 type의 [0.3]을 초기값으로 가짐.
# bias 값을 넣을 tf.Variable 생성. tf.float32 type의 [-0.3]을 초기값으로 가짐.
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# x가 tf.float32 type을 데이터로 갖는 placeholder 노드를 참조(dictionary를 feed 해주어야 함)
# y가 tf.float32 type을 데이터로 갖는 placeholder 노드를 참조(dictionary를 feed 해주어야 함)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# hypothesis 정의 : H = X * W + b
hypothesis = x * W + b

# cost function 정의
cost = tf.reduce_sum(tf.square(hypothesis - y))

# learning_rate를 0.01(강의 내에서의 알파 값)으로 두어 Gradient Descent을 적용, cost function을 minimize하도록 설정.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # tf.Variable을 이용하기 위한 초기화 작업
    sess.run(tf.global_variables_initializer())

    # 1000회 반복
    for step in range(1000):
        # train 노드를 Session을 이용하여 run. {x: x_train, y: y_train}를 feed.
        sess.run(train, {x: x_train, y: y_train})

    # 학습 결과를 확인
    # W_val, b_val, cost_val가 각각 W, b, cost가 참조하는 노드의 데이터를 참조. {x: x_train, y: y_train}를 feed.
    W_val, b_val, cost_val = sess.run([W, b, cost], feed_dict={x: x_train, y: y_train})
    # W_val, b_val, cost_val 출력
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")

"""
W: [-0.9999969] b: [0.9999908] cost: 5.699973826267524e-11
"""