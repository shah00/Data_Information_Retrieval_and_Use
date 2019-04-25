"""
Lab 6 Softmax Classifier
    lab-06-1-softmax_classifier
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# x_data 정의. shape : (8, 4)
# y_data 정의. shape : (8, 3)
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# X가 tf.float32 type의 데이터(shape : [None, 4])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
# Y가 tf.float32 type의 데이터(shape : [None, 3])를 갖는 placeholder를 참조(dictionary를 feed 해주어야 함)
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [4, 3]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [3]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# hypothesis 정의 : softmax function 사용
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost function 정의 : Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.1을 주어 cost function을 최소화 하도록 유도. W와 b를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # sess.run()으로 위의 Variable들을 초기화
    sess.run(tf.global_variables_initializer())
    # 2001회 반복
    for step in range(2001):
        # {X: x_data, Y: y_data}를 feed하여 Session을 통해 optimizer, cost 노드를 run.
        # cost_val는 갱신되는 cost 노드의 데이터를 참조
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

        # 200회 마다 회차(step), cost_val를 출력
        if step % 200 == 0:
            print(step, cost_val)

    # {X: [[1, 11, 7, 9]]}을 feed하여 Session을 통해 hypothesis 노드를 run.
    # 각 행의 element(확률) 중 최대값에 해당하는 인덱스들을 출력
    print('--------------')
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    # {X: [[1, 3, 4, 3]]}을 feed하여 Session을 통해 hypothesis 노드를 run.
    # 각 행의 element(확률) 중 최대값에 해당하는 인덱스들을 출력
    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    # {X: [[1, 1, 0, 1]]}을 feed하여 Session을 통해 hypothesis 노드를 run.
    # 각 행의 element(확률) 중 최대값에 해당하는 인덱스들을 출력
    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    # {X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]}을 feed하여 Session을 통해 hypothesis 노드를 run.
    # 각 행의 element(확률) 중 최대값에 해당하는 인덱스들을 출력
    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

    # argmax 설명 : 아래의 array는 all이 참조하는 hypothesis 노드의 데이터
    #     [[1.3890432e-03 9.9860197e-01 9.0612402e-06]
    #      [9.3119204e-01 6.2902056e-02 5.9058843e-03]
    #      [1.2732767e-08 3.3411323e-04 9.9966586e-01]]
    #         0번 행 내 최대값의 인덱스는 1
    #         1번 행 내 최대값의 인덱스는 0
    #         2번 행 내 최대값의 인덱스는 2
    #         따라서 argmax(all, 1)은 [1 0 2]
    #         argmax의 2번 째 parameter(axis)에 관한 설명 :
    #             all이 만약 vector(rank 1)이면 axis = 0을 사용해서 vector의 element 중 최대값의 인덱스를 추출할 수 있지만,
    #             all이 matrix(rank 2)이고 axis = 0일 때에는 각 열로부터 최대값의 인덱스를 추출하며
    #             all이 matrix(rank 2)이고 axis = 1일 때에는 각 행으로부터 최대값의 인덱스를 추출한다. 즉,
    #             위의 tf.argmax(all, 1)에서 axis = 1이 아닌 axis = 0을 사용했다면,
    #                 각 행이 아닌 열에서 최대값의 인덱스를 찾아올 것이다. (이 예제에서는 axis가 0일 때와 1일 때의 값이 같음.)

'''
0 6.926112
200 0.6005015
400 0.47295815
600 0.37342924
800 0.28018373
1000 0.23280522
1200 0.21065344
1400 0.19229904
1600 0.17682323
1800 0.16359556
2000 0.15216158
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
-------------
[[0.9311919  0.06290216 0.00590591]] [0]
-------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902197e-02 5.9059085e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]
'''
