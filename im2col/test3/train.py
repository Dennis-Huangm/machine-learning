import numpy as np
import model as ml
import optimizer
import tools 
from tqdm import tqdm
import pickle

data = np.load("train_data.npy")
label = np.load("train_label.npy")


label = tools.one_hot(10,label)      
data = tools.normalize(data)
data = data.reshape(-1,1,28,28)
train_data,val_data,train_label,val_label = tools.split_data(data,label,split_size=0.9)

network = ml.CNN()

epoch = 20
lr = 0.00001 #0.00005 #0.001
batch_size = 100

# opt = optimizer.Adam(lr)
opt = optimizer.Momentum(lr)
# opt = optimizer.AdaGrad(lr)

# with open("./weight/0.9218.pkl","rb") as file:
with open("./weight_dp/0.9252.pkl","rb") as file:
    params = pickle.load(file)
    file.close()
network.loadparams(params)

predict_val = network.forward(val_data)
acc0 = network.accuracy(predict_val,val_label)
print(f"=============== Test Accuracy:{acc0} ===============")

for i in range(epoch):
    iteration = int(train_data.shape[0]/batch_size)
    tqdm_bar = tqdm(range(iteration))
    
    for j in tqdm_bar:
        index_s = j*batch_size
        index_e = index_s + batch_size
        train_data_batch = train_data[index_s:index_e]
        train_label_batch = train_label[index_s:index_e]
            
        out_x = network.forward(train_data_batch)
        grads = network.gradient(train_label_batch)
        loss = network.layers[-1].loss
        opt.update(network.params,grads)

        tqdm_bar.set_description(f"epoch:{i+1}  loss:{loss} lr:{lr}")
        
    predict_val = network.forward(val_data)
    acc1 = network.accuracy(predict_val,val_label)
    print(f"=============== Test Accuracy:{acc1} ===============")

    if acc1>=acc0 and abs(acc1-acc0)<0.0001:
        lr = lr/2
        opt.lr = lr
    acc0 = acc1
    with open(f"./weight_dp/{acc1}.pkl",'wb') as file:
        pickle.dump(network.params,file)
        file.close()
