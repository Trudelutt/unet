import matplotlib.pyplot as plt
from unet import unet
from preprossesing import get_training_data
from keras.utils import to_categorical

model = unet()
train, label = get_training_data()

x_train    = train[:100]
x_validate = train[100: ]
y_train    = label[:100]
y_validate = label[100: ]

y_train_one_hot = to_categorical(y_train, num_classes=2, dtype='float32')
y_validate_one_hot = to_categorical(y_validate, num_classes=2, dtype='float32')
print(x_train.shape)
print(x_validate.shape)
#print(model.input_size)
new_x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2], 1).astype('float32')
new_y_train= y_train.reshape(y_train.shape[0],y_train.shape[1], y_train.shape[2], 1).astype('float32')
new_x_validate = x_validate.reshape(x_validate.shape[0],x_validate.shape[1], x_validate.shape[2], 1).astype('float32')
print("HER")
print(new_x_train.shape)
model.fit(x=new_x_train, y=new_y_train, batch_size=1, epochs=1, verbose=1)

p = model.predict(new_x_validate)

print(p)
print(p.shape)
new_p = p.reshape(p.shape[0], p.shape[1],p.shape[2])
plt.imshow(new_p[-1])
#plt.imshow(y_validate[0])
print(y_validate[0].shape)
plt.show()
