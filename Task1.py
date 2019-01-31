import tensorflow as tf

# build computational graph
t1 = tf.placeholder(tf.int16)
t2 = tf.placeholder(tf.int16)

addition = tf.add(t1,t2)

#initializa variable


#create a session 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Addition of t1 and t2 %i" %sess.run(addition,feed_dict={t1:2,t2:3}))


#close session
sess.close()