from DanTF_DeepNet import DanTFDeepNet
from DanTF_DeepNet import MnistData
from DanTF_DeepNet import NetLayer
from DanTF_DeepNet import LayerType

from tensorflow.examples.tutorials.mnist import input_data

layers = []

layers.append( NetLayer(layer_type=LayerType.reshape, shape=[-1,28,28,1]) )

layers.append( NetLayer(layer_type=LayerType.conv, shape=[5,5,1,32], strides=[1, 1, 1, 1]) )
layers.append( NetLayer(layer_type=LayerType.maxpool, strides=[1, 2, 2, 1], k_size=[1, 2, 2, 1]) )

layers.append( NetLayer(layer_type=LayerType.conv, shape=[5,5,32,64],strides=[1, 1, 1, 1])  )
layers.append( NetLayer(layer_type=LayerType.maxpool, strides=[1, 2, 2, 1], k_size=[1, 2, 2, 1]) )

layers.append( NetLayer(layer_type=LayerType.reshape, shape=[-1,7*7*64]) )

layers.append( NetLayer(layer_type=LayerType.fully_connected, shape=[7*7*64,1024]) )

layers.append( NetLayer(layer_type=LayerType.dropout) )

layers.append( NetLayer(layer_type=LayerType.fully_connected, shape=[1024,10]) )


mnist = input_data.read_data_sets("/mnist/", one_hot=True)
mnist_data = MnistData(mnist)

print(mnist_data.NextTrainBatch(10))

# tf_net = DanTFDeepNet(layers, mnist_data)


# tf_net.StartSession()


# tf_net.Train(1,20000,50)

# print(tf_net.Test())

# #tf_net.SaveSess("mnist-C-MP-C-MP-FCL-FCL")

# # tf_net.LoadSess("mnist-C-MP-C-MP-FCL-FCL")

# run_x, run_y = tf_net.data.NextTrainBatch(1)

# print("Prediction:") 
# print(tf_net.sess.run(tf_net.Run(run_x))[0])
# print("Actual:")
# print(list(run_y[0]).index(1))