"""
Lab 4 Multi-variable linear regression
    lab-04-2-multi_variable_matmul_linear_regression
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x_data 정의. shape : [5, 3]
# y_data 정의. shape : [5, 1]
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# X가 shape : [None, 3]이며, tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# Y가 shape : [None, 1]이며, tf.float32 type의 데이터를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [3, 1]인 대상(Matrix)를 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의 : H = XW + b
hypothesis = tf.matmul(X, W) + b

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘을 적용.
# #learning_rate에 1e-5를 주어 cost function을 최소화 하도록 유도. W와 bias를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# session 생성
sess = tf.Session()
# tf.Variable을 이용하기 위한 초기화 작업
sess.run(tf.global_variables_initializer())

# 2001회 반복
for step in range(2001):
    # {X: x_data, Y: y_data}를 feed한 cost, hypothesis, train 노드를 Session을 통해 run.
    # 갱신된 cost, hypothesis 데이터를 cost_val, hy_val이 참조
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    # 10회마다 회차(step), cost_val, hy_val을 출력
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

'''
0 Cost:  7105.46
Prediction:
 [[ 80.82241058]
 [ 92.26364136]
 [ 93.70250702]
 [ 98.09217834]
 [ 72.51759338]]
10 Cost:  5.89726
Prediction:
 [[ 155.35159302]
 [ 181.85691833]
 [ 181.97254944]
 [ 194.21760559]
 [ 140.85707092]]
...
1990 Cost:  3.18588
Prediction:
 [[ 154.36352539]
 [ 182.94833374]
 [ 181.85189819]
 [ 194.35585022]
 [ 142.03240967]]
2000 Cost:  3.1781
Prediction:
 [[ 154.35881042]
 [ 182.95147705]
 [ 181.85035706]
 [ 194.35533142]
 [ 142.036026  ]]
'''