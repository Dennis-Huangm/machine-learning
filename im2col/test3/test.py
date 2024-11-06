import numpy as np
import tools 
import pickle
import csv
import model as ml1
import model1 as ml2
from tqdm import tqdm

# network = ml1.CNN()
network1 = ml2.CNN()
network2 = ml1.CNN()

dataset = np.load("test_data.npy")
dataset = tools.normalize(dataset)
dataset = dataset.reshape(-1,1,28,28)

with open("./weight/0.9236.pkl","rb") as file1:
    params1 = pickle.load(file1)
    file1.close()
with open("./weight_dp/0.9252.pkl","rb") as file2:
    params2 = pickle.load(file2)
    file2.close()

network1.loadparams(params1)
network2.loadparams(params2)

total = range(int(dataset.shape[0]))
tqdm_bar = tqdm(total)
i=1

with open("save.csv",'w') as savefile:
    writer = csv.writer(savefile)
    writer.writerow(['Id','Category'])
    # for t in range(total):
    for t in tqdm_bar:
        data_ = np.expand_dims(dataset[t],axis=0)
        out1 = network1.forward(data_)
        out2 = network2.forward(data_)
        maxindex = np.argmax(out1+out2,axis=1)

        writer.writerow([t,maxindex[0]])
        tqdm_bar.set_description(f"number:{i}")
        i+=1
