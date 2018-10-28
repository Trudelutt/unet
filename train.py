import matplotlib.pyplot as plt
import numpy as np
from unet import unet
from preprossesing import get_training_data
from keras.utils import to_categorical

model = unet()
train, label = get_training_data()
#one_hot_label = to_categorical(label, num_classes=2, dtype=np.bool)

#x_train    = train[:100]
#x_validate = train[100: ]
#y_train    = label[:100]
#y_validate = label[100: ]

#y_train_one_hot = to_categorical(y_train, num_classes=2, dtype='float32')
#y_validate_one_hot = to_categorical(y_validate, num_classes=2, dtype='float32')
#print(x_train.shape)
#print(x_validate.shape)
#print(model.input_size)
#new_x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2], 2).astype('float32')
#new_y_train= y_train.reshape(y_train.shape[0],y_train.shape[1], y_train.shape[2], 2).astype('float32')
#new_x_validate = x_validate.reshape(x_validate.shape[0],x_validate.shape[1], x_validate.shape[2], 2).astype('float32')
#print("HER")
def visulize_predic(p):
    #p[argmax(p[....,0], p[....,1]) == p[....,0]] = 0
    new_p = np.zeros((p.shape[0], p.shape[1]))
    print("NEW P")
    print(new_p.shape)
    for i in range(new_p.shape[0]):
        for j in range(new_p.shape[1]):
            if(p[i][j][0] < p[i][j][1]):
                new_p[i][j] = 1
    return new_p


new_x_train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
y_label = label.reshape(label.shape[0], label.shape[1], label.shape[2], 1)
print(new_x_train)
print(y_label)
model.fit(x=new_x_train, y= y_label, batch_size=1, epochs=80, verbose=1)

#train1 = new_x_train[60].reshape()
p = model.predict(new_x_train)
print("Predictions")
#print(p)
print(p)
print(p.shape)
#new_p = visulize_predic(p[0])
new_p = p[0]
plt.figure()
plt.imshow(new_p[...,0])
plt.figure()
plt.imshow(train[0])
plt.figure()
#print(train[0])
plt.imshow(label[0])

#print(y_validate[0].shape)
plt.show()
