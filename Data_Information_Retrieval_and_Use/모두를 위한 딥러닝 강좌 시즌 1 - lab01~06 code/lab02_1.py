"""
Lab 2 Linear Regression
    lab-02-1-linear_regression
"""

# tensorflow를 tf란 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x_train 정의. shape : (3)
# y_train 정의. shape : (3)
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis 정의 : H = W * x + b
hypothesis = x_train * W + b

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# learning_rate를 0.01(강의 내에서의 알파 값)으로 두어 Gradient Descent을 적용, cost function을 minimize하도록 설정.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # tf.Variable을 이용하기 위한 초기화 작업
    sess.run(tf.global_variables_initializer())

    # 2001회 반복
    for step in range(2001):
        # 학습 적용. cost_val, W_val, b_val이 각각 변화되는 cost function, W(weight), b(bias)값을 참조
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        # 20회마다 step, cost_val, W_val, b_val 출력
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

# Learns best fit W:[ 1.],  b:[ 0.]
"""
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
"""
