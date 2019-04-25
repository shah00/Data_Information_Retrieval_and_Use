"""
Lab 6 Softmax Classifier
    lab-06-2-softmax_zoo_classifier
"""

# tensorflow를 tf라는 이름으로 import
# numpy를 np라는 이름으로 import
import tensorflow as tf
import numpy as np

# random seed : 777 배정
tf.set_random_seed(777)

# data-04-zoo.csv 파일을 list 형태로 불러옴. 파일 내 element type은 np.float32로, delimiter는 ','로 설정
# x_data 정의 : 마지막 열을 제외한 나머지
# y_data 정의 : 마지막 열
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# x_data와 y_data의 shape를 출력
print(x_data.shape, y_data.shape)
'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6

# X가 tf.float32 type의 데이터(shape : [None, 16])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# Y가 tf.int32 type의 데이터(shape : [None, 1])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# one_hot 예시 : indices = [0, 1, 2], depth = 3
#     tf.one_hot(indices, depth)  # output: [3 x 3]
#         [[1., 0., 0.],
#          [0., 1., 0.],
#          [0., 0., 1.]]        출처 : https://www.tensorflow.org/api_docs/python/tf/one_hot

# Y array를 one_hot함수를 통해 변환하고, 이를 Y_one_hot이 참조
# Y_one_hot의 shape를 출력 : (?, 1, 7)
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot:", Y_one_hot)
# rank가 2에서 3으로 변환되었으므로, 다시 되돌리기 위해 reshape. reshape한 Y_one_hot의 shape를 출력 : (?, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)
'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [16, 7]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [7]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# logits 정의 : linear regression에서 사용한 hypothesis = XW + b
# hypothesis 정의 : softmax function 사용. logits을 대입.
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# cost function 정의 : cross-entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                  labels=tf.stop_gradient([Y_one_hot])))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.1을 주어 cost function을 최소화 하도록 유도. W와 b를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# *** argmax 함수에 대한 설명은 lab06_1을 참조 ***
# argmax 함수를 이용하여 hypothesis 내 각 행의 element 중 최대값의 index들을 찾고,
#     이를 prediction이 array로 참조(정확히는 해당 array를 데이터로 갖는 노드)
# correct_prediction은 Y_one_hot 내 각 행의 element 중
#     최대값의 index들을 element로 갖는 array와 prediction을 비교한 진리값들을 element로 갖는 array(정확히는 데이터로 갖는 노드)
# accuracy는 correct_prediction의 True, False 값을 1과 0으로 casting한 다음, 평균낸 값을 참조(정확히는 데이터로 갖는 노드)
# 예시) hypothesis : [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]] -> prediction : [0, 2]
#       tf.argmax(Y_one_hot, 1)이 [0, 1]이라고 가정한다면
#       correct_prediction은 [True, False]
#       tf.cast(correct_prediction, tf.float32)는 correct_prediction을 cast한 array(True는 1., False는 0.으로)
#       accuracy는 casting된 element들의 평균 : (0.5)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # sess.run()으로 위의 Variable들을 초기화
    sess.run(tf.global_variables_initializer())

    # 2001회 반복
    for step in range(2001):
        # {X: x_data, Y: y_data}를 feed하여 Session을 통해 optimizer, cost, accuracy 노드를 run.
        # cost_val, acc_val은 갱신된 cost, accuracy 노드의 데이터를 참조
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
        # 100회마다 회차(step), cost_val, acc_val 출력
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # {X: x_data}를 feed하여 prediction 노드를 run. pred가 갱신된 데이터를 참조
    pred = sess.run(prediction, feed_dict={X: x_data})

    # flatten 함수는 Multidimensional array를 One-dimensional array로 변환해주는 함수
    # zip 함수는 element 수가 동일한 iterable object를 묶어주는 함수
    # p는 pred를, y는 변환된 y_data를 참조
    # 두 array의 element들이 같은지 비교하여 출력
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
...
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''
