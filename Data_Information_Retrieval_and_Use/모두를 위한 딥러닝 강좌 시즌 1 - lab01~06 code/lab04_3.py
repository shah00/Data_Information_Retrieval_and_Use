"""
Lab 4 Multi-variable linear regression
    lab-04-3-file_input_linear_regression
"""

# tensorflow를 tf라는 이름으로 import
# numpy를 np라는 이름으로 import
import tensorflow as tf
import numpy as np

# random seed : 777 배정
tf.set_random_seed(777)

# data-01-test-score.csv 파일을 list 형태로 불러옴. 파일 내 element type은 np.float32로, delimiter는 ','로 설정
# x_data 정의 : 마지막 열을 제외한 나머지
# y_data 정의 : 마지막 열
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# x, y의 shape를 출력
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# data output
'''
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 ...
 [ 76.  83.  71.]
 [ 96.  93.  95.]] 
x_data shape: (25, 3)
[[152.]
 [185.]
 ...
 [149.]
 [192.]] 
y_data shape: (25, 1)
'''

# X가 tf.float32 type의 데이터(shape : [None, 3])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# Y가 tf.float32 type의 데이터(shape : [None, 1])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [3, 1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의 : XW + b
hypothesis = tf.matmul(X, W) + b

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘을 적용. learning_rate에 1e-5를 주어 cost function을 최소화 하도록 유도. W, b를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Session 생성
sess = tf.Session()
# tf.Variable을 이용하기 위한 초기화 작업
sess.run(tf.global_variables_initializer())

# 2001회 반복
for step in range(2001):
    # {X: x_data, Y: y_data}를 feed한 cost, hypothesis, train 노드를 Session을 통해 run.
    # 갱신된 cost, hypothesis 데이터를 cost_val, hy_val이 참조
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    # 10회마다 cost_val, hy_val 출력
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

# train output
'''
0 Cost: 21027.0 
Prediction:
 [[22.048063 ]
 [21.619772 ]
 ...
 [31.36112  ]
 [24.986364 ]]
10 Cost: 95.976326 
Prediction:
 [[157.11063 ]
 [183.99283 ]
 ...
 [167.48862 ]
 [193.25117 ]]
 1990 Cost: 24.863274 
Prediction:
 [[154.4393  ]
 [185.5584  ]
 ...
 [158.27443 ]
 [192.79778 ]]
2000 Cost: 24.722485 
Prediction:
 [[154.42894 ]
 [185.5586  ]
 ...
 [158.24257 ]
 [192.79166 ]]
'''
# Session을 통해 {X: [[100, 70, 101]]}를 feed하여 hypothesis 노드를 run, 갱신된 값을 출력
print("Your score will be ", sess.run(hypothesis,
                                      feed_dict={X: [[100, 70, 101]]}))

# Session을 통해 {X: [[60, 70, 110], [90, 100, 80]]}를 feed하여 hypothesis 노드를 run, 갱신된 값을 출력
print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]
'''
