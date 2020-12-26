#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from keras.datasets import cifar100
import matplotlib as plt
from matplotlib.figure import Figure
from matplotlib import figure
from keras.callbacks import Callback
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from scikitplot.metrics import plot_confusion_matrix
from keras.optimizers import SGD
from keras.layers import Dropout


def prep_pixels(trainX, testX):
    trainNorm = trainX.astype('float32')
    testNorm = testX.astype('float32')
    trainNorm = trainNorm / 255.0
    testNorm = testNorm / 255.0
    return trainNorm, testNorm

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='orange', label='train')
    
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        return
    
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = to_categorical(trainY)
testY = to_categorical(testY)
trainX, testX = prep_pixels(trainX, testX)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=1)
model = define_model()
metrics = Metrics()
history = model.fit(trainX, trainY, epochs=100, batch_size=64,  validation_split = 0.2, verbose=2, callbacks=[metrics])
_, acc = model.evaluate(valX, valY, verbose=2)
summarize_diagnostics(history)

model.summary()
predY = model.predict(testX)
predYClasses = predY.argmax(1)
testYClasses = testY.argmax(1)
print(accuracy_score(testYClasses, predYClasses))
print(f1_score(testYClasses, predYClasses, average='macro'))
print(recall_score(testYClasses, predYClasses, average='macro'))
print(precision_score(testYClasses, predYClasses, average='macro'))


# In[1]:


(trainXCoarse, trainYCoarse), (testXCoarse, testYCoarse) = cifar100.load_data('coarse')
(trainXFine, trainYFine), (testXFine, testYFine) = cifar100.load_data('fine')
trainXClass = trainXFine[(trainYCoarse == 14).squeeze()]
trainYClass = trainYFine[(trainYCoarse == 14).squeeze()]
trainYClass[trainYClass==[11]] = 10
trainYClass[trainYClass==[98]] = 11
trainYClass[trainYClass==[35]] = 12
trainYClass[trainYClass==[46]] = 13
trainYClass[trainYClass==[2]] = 14
testXClass = testXFine[(testYCoarse == 14).squeeze()]
testYClass = testYFine[(testYCoarse == 14).squeeze()]
testYClass[testYClass==[11]] = 10
testYClass[testYClass==[98]] = 11
testYClass[testYClass==[35]] = 12
testYClass[testYClass==[46]] = 13
testYClass[testYClass==[2]] = 14
trainY_classes = trainY.argmax(1)
testY_classes = testY.argmax(1)
trainXClass, testXClass = prep_pixels(trainXClass, testXClass)
trainXAll = np.concatenate((trainX, trainXClass))
testXAll = np.concatenate((testX, testXClass))
trainYAll = np.concatenate((trainY_classes, trainYClass))
testYAll = np.concatenate((testY_classes, testYClass))
trainYAll = to_categorical(trainYAll)
testYAll = to_categorical(testYAll)
plt.figure.Figure(figsize=(7,7))
    
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(trainXClassv[i])
pyplot.show()

for layer in model.layers: 
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')
    
last = model.layers[-1].output
x = Dense(1000, activation='relu')(last)
x = Dropout(0.3)(x)
x = Dense(15, activation='softmax')(x)
newModel = Model(model.input, x)
newModel.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
newModel.summary()


# In[2]:


trainXAll, valXClass, trainYAll, valYClass = train_test_split(trainXAll, trainYAll, test_size=0.25, random_state=1)
newMetrics = Metrics()
newHistory = newModel.fit(trainXAll, trainYAll, epochs=100, batch_size=64,  validation_split = 0.2, verbose=2, callbacks=[newMetrics])
_, acc = newModel.evaluate(valXClass, valYClass, verbose=2)
summarize_diagnostics(newHistory)


# In[ ]:


predYAll = new_model.predict(testXAll)
predYAllClasses = predYAll.argmax(1)
testYAll = testYAll.argmax(1)
print(accuracy_score(testYAll, predYAllClasses))
print(f1_score(testYAll, predYAllClasses, average='macro'))
print(recall_score(testYAll, predYAllClasses, average='macro'))
print(precision_score(testYAll, predYAllClasses, average='macro'))

