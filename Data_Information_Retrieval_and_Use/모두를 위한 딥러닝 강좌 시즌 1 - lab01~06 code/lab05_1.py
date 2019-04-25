"""
Lab 5 Logistic Regression Classifier
    lab-05-1-logistic_regression
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x_data 정의. shape : [6, 2]
# y_data 정의. shape : [6, 1]
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# X가 tf.float32 type의 데이터(shape : [None, 2])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# Y가 tf.float32 type의 데이터(shape : [None, 1])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [2, 1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의(activation function으로 sigmoid 함수를 사용) : 1 / (1 + e^-(XW + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# logistic regression을 위해 사용하는 cost function 정의
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.01을 주어 cost function을 최소화 하도록 유도. W, b를 조정해줌.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# hypothesis(가설) 값이 0.5보다 크면 1로, 작거나 같으면 0으로 casting. 결과를 predicted가 참조
# accuracy(정확도) 측정. 예측값(1 혹은 0으로 변환된 값)과 실제 값이 같은 경우 1, 다른 경우 0으로 둔 다음 평균을 냄
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # sess.run()으로 위의 Variable들을 초기화
    sess.run(tf.global_variables_initializer())

    # 10001회 반복
    for step in range(10001):
        # {X: x_data, Y: y_data}를 feed, Session을 통해 cost, train 노드를 run. cost_val은 갱신된 cost 노드의 데이터를 참조
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        # 200회마다 회차(step), cost_val 출력
        if step % 200 == 0:
            print(step, cost_val)

    # {X: x_data, Y: y_data}를 feed, Session을 통해 hypothesis, predicted, accuracy 노드를 run
    # h, c, a가 각각 최종적으로 변화된 hypothesis, predicted, accuracy 노드의 데이터를 참조
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496
Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''