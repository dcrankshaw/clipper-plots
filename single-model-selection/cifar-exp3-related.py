import numpy as np
import json
import matplotlib.pyplot as plt


### cifar10 exp3 for single model selection
original_acc = [0.7575, 0.654, 0.832, 0.54, 0.9085]
avg_ac = json.load(open('../data/cifar-exp3-acc.json', 'r'))
fig, ax = plt.subplots()
ax.plot(range(len(avg_ac)), avg_ac, label='exp3')
for i in range(5):
    ax.plot(range(len(avg_ac)), np.ones((len(avg_ac),))*original_acc[i], '--',label='model'+str(i+1))
plt.xlabel('queries')
plt.ylabel('accuracy')
ax.set_ylim([0.5,1])
ax.legend(loc=4)
plt.savefig('cifar10-exp3.pdf')

### failure in model5 
original_acc = [0.7575, 0.654, 0.832, 0.54, 0.9085]
avg_drift = json.load(open('../data/cifar-exp3-drift.json', 'r'))
fig, ax = plt.subplots()
ax.plot(range(len(avg_drift)), avg_drift, label='exp3-drift')
for i in range(5):
    ax.plot(range(len(avg_drift)), np.ones((len(avg_drift),))*original_acc[i], '--',label='model'+str(i+1))
plt.axvline(x=1000,color='black',linestyle='--')
ax.annotate('failure in model5', xy=(1001, 0.9), xytext=(1300, 0.92),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.xlabel('queries')
plt.ylabel('accuracy')
ax.set_ylim([0.5,1])
ax.legend(loc=4)
plt.savefig('cifar10-exp3-drift.pdf')
