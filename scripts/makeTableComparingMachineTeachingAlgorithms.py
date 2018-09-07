import numpy as np
import matplotlib.pyplot as plt
from numpy import nan, inf

def to_table_row(data_array):
        row = ""
        for i in range(len(data_array)):
            if i < (len(data_array) - 1):
                separator = " & "
            else:
                separator = " \\\\"
                
            if np.round(data_array[i]) == data_array[i]:
                row += '{0:.1f}'.format(data_array[i]) + separator
            else:
                row += '{0:.3f}'.format(data_array[i]) + separator
        return row


filePath = "./data/algorithm_comparison/"
width = 9  #9x9 grid world
numFeatures = 8



samples = [10e4, 10e5, 10e6]
for s in samples:
    #print(s)
    filename = "cakmak" + "_N=" + str(width) + "_F=" + str(numFeatures) + "_samp" + str(int(s)) + "_" 
    all_data = []
    f = open(filePath + filename + "random_trajLength=1.txt",'r')
    times = []
    demos = []
    losses = []
    zero_one_loss = []
    num_sa_pairs = []
    for line in f:
        data = line.strip().split(",")
        times.append(float(data[0]))
        num_sa_pairs.append(float(data[2]))
        zero_one_loss.append(float(data[4]))

    #all_data.append(np.mean(demos)) 
    all_data.append(np.mean(num_sa_pairs))
    all_data.append(np.mean(zero_one_loss))   
    all_data.append(np.mean(times))
    #print all_data
    print("UVM (" + str(int(s)) + ") &",to_table_row(all_data))
            

# #print(s)
# filename = "baseline__" 
# all_data = []
# f = open(filePath + filename + "random_trajLength=1.txt",'r')
# times = []
# demos = []
# losses = []
# zero_one_loss = []
# num_sa_pairs = []
# for line in f:
#     data = line.strip().split(",")
#     times.append(float(data[0]))
#     demos.append(float(data[1]))
#     losses.append(float(data[2]))
#     zero_one_loss.append(float(data[3]))
#     num_sa_pairs.append(float(data[4]))
# #all_data.append(np.mean(demos)) 
# all_data.append(np.mean(num_sa_pairs))
# all_data.append(np.mean(losses))
# all_data.append(np.mean(zero_one_loss))   
# all_data.append(np.mean(times))
# #print all_data
# print("RCC &",to_table_row(all_data))


filename = "scot" 
all_data = []
f = open(filePath + filename + "_N=" + str(width) + "_F=" + str(numFeatures) + "_random_trajLength=1.txt",'r')
times = []
demos = []
losses = []
zero_one_loss = []
num_sa_pairs = []
for line in f:
    data = line.strip().split(",")
    times.append(float(data[0]))
    num_sa_pairs.append(float(data[2]))
    zero_one_loss.append(float(data[4]))
    
#all_data.append(np.mean(demos)) 
all_data.append(np.mean(num_sa_pairs))
all_data.append(np.mean(zero_one_loss))   
all_data.append(np.mean(times))
#print all_data
print("SCOT &",to_table_row(all_data))


# filename = "non-redundant__" 
# all_data = []
# f = open(filePath + filename + "random_trajLength=1.txt",'r')
# times = []
# demos = []
# losses = []
# zero_one_loss = []
# num_sa_pairs = []
# for line in f:
#     data = line.strip().split(",")
#     times.append(float(data[0]))
#     demos.append(float(data[1]))
#     losses.append(float(data[2]))
#     zero_one_loss.append(float(data[3]))
#     num_sa_pairs.append(float(data[4]))
# #all_data.append(np.mean(demos)) 
# all_data.append(np.mean(num_sa_pairs))
# all_data.append(np.mean(losses))
# all_data.append(np.mean(zero_one_loss))   
# all_data.append(np.mean(times))
# #print all_data
# print("NRCC &",to_table_row(all_data))


# filename = "random_samples__" 
# all_data = []
# f = open(filePath + filename + "random_trajLength=1.txt",'r')
# times = []
# demos = []
# losses = []
# zero_one_loss = []
# num_sa_pairs = []
# for line in f:
#     data = line.strip().split(",")
#     times.append(float(data[0]))
#     demos.append(float(data[1]))
#     losses.append(float(data[2]))
#     zero_one_loss.append(float(data[3]))
#     num_sa_pairs.append(float(data[4]))
# #all_data.append(np.mean(demos)) 
# all_data.append(np.mean(num_sa_pairs))
# all_data.append(np.mean(losses))
# all_data.append(np.mean(zero_one_loss))   
# all_data.append(np.mean(times))
# #print all_data
# print("Random &",to_table_row(all_data))


