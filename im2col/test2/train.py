import numpy as np
import model
import optimizer
import tools 
from tqdm import tqdm
import pickle

# 数据集导入
data = np.load("../train_data.npy")
label = np.load("../train_label.npy")


label = tools.one_hot(10,label)      
data = tools.normalize(data)
data = data.reshape(-1,1,28,28)
print(data.shape)
train_data,val_data,train_label,val_label = tools.split_data(data,label,split_size=0.9)

# 导入模型
network = model.CNN()
network1 = model.CNN()
network2 = model.CNN()
network3 = model.CNN()
network4 = model.CNN()
network5 = model.CNN()

# 超参数
epoch = 10
lr = 0.001
batch_size = 100


# 优化器
opt = optimizer.Adam(lr)

# 加载模型
# with open("./weight/0.8988.pkl","rb") as file:
#     params = pickle.load(file)
#     file.close()
# network1.loadparams(params)

predict_val = network1.forward(val_data)
acc0 = network1.accuracy(predict_val,val_label)
print(f"=============== Test Accuracy:{acc0} ===============")

for i in range(epoch):
        
    iteration = int(train_data.shape[0]/batch_size)
        
    tqdm_bar = tqdm(range(iteration))
        
    for j in tqdm_bar:
        index_s = j*batch_size
        index_e = index_s + batch_size
        #train_data_batch = train_data[index_s:index_e]
        #train_label_batch = train_label[index_s:index_e]
        train_data_batch = train_data[index_s:index_e]
        train_label_batch = train_label[index_s:index_e]
            
        out_x = network.forward(train_data_batch)
        grads = network.gradient(train_label_batch)
        loss = network.layers[-1].loss
        opt.update(network.params,grads)
            
        tqdm_bar.set_description(f"epoch:{i+1}  loss:{loss}")
        
    predict_val = network.forward(val_data)
    acc1 = network.accuracy(predict_val,val_label)
    print(f"=============== Test Accuracy:{acc1} ===============")

    if acc1>=acc0 and abs(acc1-acc0)<0.0001:
        lr = lr/2
        opt.lr = lr
    acc0 = acc1
    with open(f"./weight/{acc1}.pkl",'wb') as file:
        pickle.dump(network.params,file)
