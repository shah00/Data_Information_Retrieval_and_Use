"""
Lab 4 Multi-variable linear regression
    lab-04-1-multi_variable_linear_regression
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x1_data, x2_data, x3_data 정의. 3개의 feature.
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

# y_data 정의
y_data = [152., 185., 180., 196., 142.]

# x1이 tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# x2가 tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# x3가 tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

# y가 tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
Y = tf.placeholder(tf.float32)

# weight1 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight1'로 줌
# weight2 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight2'로 줌
# weight3 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight3'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의 : x1 * w1 + x2 * w2 + x3 * w3 + b
# Broadcasting 발생
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘을 적용.
# #learning_rate에 1e-5를 주어 cost function을 최소화 하도록 유도. 3개의 weight와 1개의 bias를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# session 생성
sess = tf.Session()
# tf.Variable을 이용하기 위한 초기화 작업
sess.run(tf.global_variables_initializer())

# 2001회 반복
for step in range(2001):
    # {x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data}을 feed 하고, cost, hypothesis, train 노드를 Session을 통해 run.
    # cost_val, hy_val은 각각 갱신된 cost, hypothesis 노드의 데이터를 참조
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    # 10회마다 회차(step), cost_val, hy_val 값 출력
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

'''
0 Cost:  19614.8
Prediction:
 [ 21.69748688  39.10213089  31.82624626  35.14236832  32.55316544]
10 Cost:  14.0682
Prediction:
 [ 145.56100464  187.94958496  178.50236511  194.86721802  146.08096313]
 ...
1990 Cost:  4.9197
Prediction:
 [ 148.15084839  186.88632202  179.6293335   195.81796265  144.46044922]
2000 Cost:  4.89449
Prediction:
 [ 148.15931702  186.8805542   179.63194275  195.81971741  144.45298767]
'''