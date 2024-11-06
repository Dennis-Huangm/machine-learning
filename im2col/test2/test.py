import numpy as np
import tools 
import pickle
import csv
import model
import tqdm

network = model.CNN()

dataset = np.load("test_data.npy")
dataset = tools.normalize(dataset)
dataset = dataset.reshape(-1,1,28,28)

with open("./weight/0.9016.pkl","rb") as file:
    params = pickle.load(file)
    file.close()
network.loadparams(params)
total = int(dataset.shape[0])
# tqdm_bar = tqdm(total)
with open("save.csv",'w') as savefile:
    writer = csv.writer(savefile)
    writer.writerow(['Id','Category'])
    for t in range(total):
    # for t in tqdm_bar:
        data_ = np.expand_dims(dataset[t],axis=0)
        out = network.forward(data_)
        maxindex = np.argmax(out,axis=1)
        writer.writerow([t,maxindex[0]])