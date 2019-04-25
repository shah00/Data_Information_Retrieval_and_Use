"""
Lab 5 Logistic Regression Classifier
    lab-05-2-logistic_regression_diabetes
"""

# tensorflow를 tf란 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x data 정의. shape : (6, 2)
# y data 정의. shape : (6, 1)
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

# X(Matrix)가 type은 float32, shape : (None, 2)인 placeholder 참조(dictionary를 feed 해주어야 함)
# Y(Matrix)가 type은 float32, shape : (None, 1)인 placeholder 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W가 weight으로 이름지은 shape : (2, 1)의 tf 변수 참조. 초기 값은 랜덤 생성
# b가 bias로 이름지은 shape : (1, )의 tf 변수 참조. 초기 값은 랜덤 생성
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의. (X*W + b)를 sigmoid 함수에 대입한 값을 참조
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost function 정의. Y가 1일 때와 0일 때를 모두 고려함. lab6에 등장하는 cross-entropy function과 일치함.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.01을 주어 cost function을 최소화 하도록 유도.
# W(weight Matrix)와 b를 조정해줌.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 예측 값. cast 함수는 텐서를 새로운 타입으로 casting하는 함수.
# hypothesis가 0.5보다 크면 True(1)로, 그 이외에는 False(0)으로 casting. type은 tf.float32로.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# 정확도 측정 값. 위의 predicted와 Y를 비교하여 동일하면 True(1)을,
# 다르다면 False(0)이 되도록 캐스팅한 다음 평균을 계산.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # sess.run()으로 위의 Variable들을 초기화
    sess.run(tf.global_variables_initializer())

    # 학습 진행(10001회)
    for step in range(10001):
        # train 노드에 X에 x_data를, Y에 y_data를 각각 feed한 다음 cost와 함께 실행한다(구한다). cost값은 cost_val가 참조한다.
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        # step값이 200으로 나누어 떨어질 때마다
        if step % 200 == 0:
            # step과 cost값을 출력한다.
            print(step, cost_val)

    # 학습 종료 후 hypothesis 내 X, Y에 각각 x_data, y_data를 feed한 다음
    # hypothesis, predicted, accuracy를 구하고, 이를 h, c, a가 각각 참조하게끔 만듬.
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    # 학습을 모두 마쳤으므로, hypothesis, predicted, accuracy 값을 출력
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
