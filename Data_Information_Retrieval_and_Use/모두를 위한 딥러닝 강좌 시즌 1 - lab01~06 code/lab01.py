"""
Lab 1 Basics
    lab-01-basics
"""

# Tensorflow를 tf라는 이름으로 import
import tensorflow as tf

# hello라는 variable이 "Hello, Tensorflow!"라는 상수 데이터를 가진 노드를 참조
hello = tf.constant("Hello, TensorFlow!")

# Tensorflow running을 위한 Session. sess가 참조
sess = tf.Session()

# sess을 통해 hello 노드를 run
print(sess.run(hello))

### rank,shape 개념 예시 ###
# 3 - rank : 0, shape : []인 scalar
# [1. ,2., 3.], rank : 1, shape : [3]인 vector
# [[1., 2., 3.], [4., 5., 6.]], rank : 2, shape : [2, 3]인 matrix
# [[[1., 2., 3.]], [[7., 8., 9.]]], rank : 3, shape : [2, 1, 3]인 tensor

# node1이 참조하는 것은 3.0의 상수 float32 type을 데이터로 갖는 노드
# node2이 참조하는 것은 4.0의 상수 float32 type을 데이터로 갖는 노드로 data type을 명시하지 않으면 프로그램 내에서 유추
# node3이 참조하는 것은 node1과 node2의 데이터를 합한 데이터를 갖는 노드
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

# node1과 node2의 데이터를 출력, Session을 이용하지 않았으므로, 해당 Tensor에 대한 내용이 출력
# node3의 데이터를 출력, Session을 이용하지 않았으므로, 해당 Tensor에 대한 내용이 출력
print("node1:", node1, "node2:", node2)
print("node3: ", node3)

# Tensorflow running을 위한 Session. sess가 참조
# Session을 이용하여 node1, node2를 run
# Session을 이용하여 node3을 run
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

# tf.float32의 데이터를 갖는 placeholder 노드를 a가 참조하도록 함
# 이 placeholder는 dict를 feed할 것을 요구함
a = tf.placeholder(tf.float32)

# tf.float32의 데이터를 갖는 placeholder 노드를 b가 참조하도록 함
# 이 placeholder는 dict를 feed할 것을 요구함
b = tf.placeholder(tf.float32)

# a노드와 b노드가 참조하는 데이터의 합을 참조하는 adder_node를 생성
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# adder_node에 {a: 3, b: 4.5}를 feed하여 run
# adder_node에 {a: [1, 3], b: [2, 4]}를 feed하여 run
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

# adder_node가 참조하는 데이터에 3을 곱한 데이터를 참조하는 add_and_triple노드 생성
add_and_triple = adder_node * 3
# add_and_triple에 {a: 3, b: 4.5}를 feed하여 run
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))
