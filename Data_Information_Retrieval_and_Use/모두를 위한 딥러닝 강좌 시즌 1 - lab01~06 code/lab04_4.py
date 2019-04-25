"""
Lab 4 Multi-variable linear regression
    lab-04-4-tf_reader_linear_regression
"""

# tensorflow를 tf란 이름으로 import
import tensorflow as tf

# random seed : 777 배정
tf.set_random_seed(777)

# 'data-01-test-score.csv' 파일에 저장된 문자열을 차례대로 큐(queue)에 출력.
filename_queue = tf.train.string_input_producer(
['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# reader 정의. 행을 출력.
# key, value에 출력하는 값을 할당
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# csv내 Field 의 data type을 미리 정의
# csv로 decode 하도록 명령
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# batch들을 생성
train_x_batch, train_y_batch = \
tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# 나중에 값을 받기 위해 비워둔 tensor. 받을 값은 shape=[None, 3]인 X(실 측정? 데이터 값)
# 나중에 값을 받기 위해 비워둔 tensor. 받을 값은 shape=[None, 1]인 Y(실제 결과 값)
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [3, 1]인 대상을 랜덤으로 생성하여 가짐. name은 'weight'로 줌
# bias 값을 넣을 tf.Variable 생성. 초기 값으로 shape = [1]인 대상을 랜덤으로 생성하여 가짐. name은 'bias'로 줌
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 정의 : H = XW + b
hypothesis = tf.matmul(X, W) + b
# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning_rate를 1e-5(강의 내에서의 알파 값)으로 두어 Gradient Descent을 적용, cost function을 minimize하도록 설정.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# session 생성
# tf.Variable을 이용하기 위한 초기화 작업
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 쓰레드(Thread)를 종료시키는 기능을 coord variable에 저장
# 그래프 내의 queue_runner를 시작
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 2001번 실행
for step in range(2001):
    # session내의 train_x_batch, train_y_batch를 run하여 x_batch, y_batch에 할당
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    # Cost function, hypothesis, train tensor에 X, Y 값을 feed하여 run, cost_val, hy_val, _에 할당
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    # 10회마다 Cost_function value와 Prediction value를 출력
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# 쓰레드(Thread) 중지
# 쓰레드(Thread) 조인(join)
coord.request_stop()
coord.join(threads)

# hypothesis에 X 값을 제공하여 session을 통해 run
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

# hypothesis에 X 값을 제공하여 session을 통해 run
print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[185.33531]]
Other scores will be  [[178.36246]
[177.03687]]
'''
