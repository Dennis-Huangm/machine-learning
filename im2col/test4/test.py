import numpy as np
import tools
import pickle
import csv
import model
import model2
import model3
import model4
import model5
from tqdm import tqdm

network1 = model.CNN()
network2 = model2.CNN()
network3 = model3.CNN()
network4 = model4.CNN()
network5 = model5.CNN()

dataset = np.load("test_data.npy")
dataset = tools.normalize(dataset)
dataset = dataset.reshape(-1, 1, 28, 28)

with open("./weight1/0.901.pkl", "rb") as file:
    params1 = pickle.load(file)
    file.close()
network1.loadparams(params1)

with open("./weight2/0.9.pkl", "rb") as file:
    params2 = pickle.load(file)
    file.close()
network2.loadparams(params2)

with open("./weight3/0.895.pkl", "rb") as file:
    params3 = pickle.load(file)
    file.close()
network3.loadparams(params3)

# with open("./weight4/0.931.pkl","rb") as file:
#     params4 = pickle.load(file)
#     file.close()
# network4.loadparams(params4)

# with open("./weight5/0.928.pkl","rb") as file:
#     params5 = pickle.load(file)
#     file.close()
# network5.loadparams(params5)


total = int(dataset.shape[0])

with open("save.csv", 'w') as savefile:
    writer = csv.writer(savefile)
    writer.writerow(['Index', 'ID'])
    for t in tqdm(range(total)):
        data_ = np.expand_dims(dataset[t], axis=0)
        # bagging
        out = network1.forward(data_)
        out += network2.forward(data_)
        out += network3.forward(data_)
        # out += network4.forward(data_)
        # out += network5.forward(data_)
        maxindex = np.argmax(out, axis=1)
        writer.writerow([t, maxindex[0]])

print("save compelete!")
