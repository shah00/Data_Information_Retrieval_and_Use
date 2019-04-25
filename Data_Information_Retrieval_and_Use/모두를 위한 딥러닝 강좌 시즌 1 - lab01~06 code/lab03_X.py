"""
Lab 3 Minimizing Cost
    lab-03-X-minimizing_cost_tf_gradient
"""

# tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# X, Y 값 정의
X = [1, 2, 3]
Y = [1, 2, 3]

# weight 값을 넣을 tf.Variable 생성. tf.float32 type의 5.0를 초기값으로 가짐.
W = tf.Variable(5.)

# hypothesis 정의 : H = XW (bias 생략)
hypothesis = X * W

# cost function을 미분한 값(기울기). 아래의 gvs와 달리, 직접 수식을 적용함
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost function 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 알고리즘을 적용. learning_rate에 0.01을 주어 cost function을 최소화 하도록 유도. W를 조정해줌.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# gradient(기울기)를 계산, gvs가 참조하도록 함
gvs = optimizer.compute_gradients(cost)

# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

# gvs apply
apply_gradients = optimizer.apply_gradients(gvs)

# Session을 생성하여 sess라고 명명. 예외 발생을 포함한 컨텍스트 종료 시 자동으로 세션 종료.
with tf.Session() as sess:
    # tf.Variable을 이용하기 위한 초기화 작업
    sess.run(tf.global_variables_initializer())
    # 101회 반복
    for step in range(101):
        # Session을 통해 gradient, gvs, apply_gradients 노드를 run
        # gradient_val, gvs_val은 갱신된 gradient, gvs 노드의 데이터를 참조
        gradient_val, gvs_val, _ = sess.run([gradient, gvs, apply_gradients])
        # 회차(step), gradient_val, gvs_val을 출력. gvs_val은 optimizer에 의한 gradient와 weight값을 가짐
        print(step, gradient_val, gvs_val)

'''
0 37.333332 [(37.333336, 5.0)]
1 33.84889 [(33.84889, 4.6266665)]
2 30.689657 [(30.689657, 4.2881775)]
3 27.825289 [(27.825289, 3.981281)]
...
97 0.0027837753 [(0.0027837753, 1.0002983)]
98 0.0025234222 [(0.0025234222, 1.0002704)]
99 0.0022875469 [(0.0022875469, 1.0002451)]
100 0.0020739238 [(0.0020739238, 1.0002222)]
'''