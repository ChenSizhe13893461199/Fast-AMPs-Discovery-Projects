import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 加载训练数据
train_data =...  # load training data
train_labels = ...  # load training label

# 创建模型
model = Sequential()
model.add(LSTM(units=128, input_shape=(50, 20)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32)
predictions_p = model.predict([])#fill the [] with preidction input
print(np.sum(predictions_p[990:,0]>0.5))#Print AMPs predictions
print(np.sum(predictions_p[:990,1]>0.5))#Print Non-AMPs predictions