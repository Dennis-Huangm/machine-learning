import numpy as np
import model
import model2
import model3
import model4
import model5
import optimizer
import tools
from tqdm import tqdm
import pickle

# 数据集导入
data = np.load("../train_data.npy")
label = np.load("../train_label.npy")

label = tools.one_hot(10, label)
data = tools.normalize(data)
data = data.reshape(-1, 1, 28, 28)
train_data, val_data, train_label, val_label = tools.split_data(data, label, split_size=0.9)

# 选择模型
selected_model = 4
if selected_model == 1:
    network = model.CNN()
elif selected_model == 2:
    network = model2.CNN()
elif selected_model == 3:
    network = model3.CNN()
elif selected_model == 4:
    network = model4.CNN()
else:
    network = model5.CNN()
    selected_model = 5

# 超参数
epoch = 25
lr = 0.001
batch_size = 64

# 优化器
opt = optimizer.Adam(lr)

# 加载已有权重
# with open("./weight2/0.9302.pkl","rb") as file:
#      params = pickle.load(file)
#      file.close()
# for k,v in params.items():
#     params[k] = v
#
# network.loadparams(params)

predict_val = network.forward(val_data)
acc0 = network.accuracy(predict_val, val_label)
print(f"=============== 准确率:{acc0} ===============")

for i in range(epoch):

    iteration = int(train_data.shape[0] / batch_size)

    pbar = tqdm(range(iteration))

    for j in pbar:
        index_s = j * batch_size
        index_e = index_s + batch_size
        train_data_batch = train_data[index_s:index_e]
        train_label_batch = train_label[index_s:index_e]

        out_x = network.forward(train_data_batch)
        grads = network.gradient(train_label_batch)
        loss = network.layers[-1].loss
        opt.update(network.params, grads)

        pbar.set_description(f"Epoch:{i + 1} Loss:{loss} 学习率:{lr}")

    predict_val = network.forward(val_data)
    acc1 = network.accuracy(predict_val, val_label)
    print(f"=============== 准确率:{acc1} ===============")

    if i % 5 == 0 and i != 0:
        lr = lr / 2
        opt.lr = lr
    acc0 = acc1
    with open(f"./weight{selected_model}/{acc1}.pkl", 'wb') as file:
        pickle.dump(network.params, file)
