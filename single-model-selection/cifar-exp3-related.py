import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
### cifar10 exp3 for single model selection
original_acc = [0.7575, 0.654, 0.832, 0.54, 0.9085]
avg_ac = json.load(open('./data/cifar-exp3-acc.json', 'r'))
fig, ax = plt.subplots(figsize=(4,3))
colors = sns.color_palette("bright", n_colors=6, desat=.5)
ax.plot(range(len(avg_ac)), avg_ac, color=colors[0],label='exp3')
for i in range(5):
    ax.plot(range(len(avg_ac)), np.ones((len(avg_ac),))*original_acc[i], '--', color=colors[i+1],label='model'+str(i+1))
plt.xlabel('queries', fontsize=15)
plt.ylabel('accuracy', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.set_ylim([0.5,1])
ax.legend(loc=4, frameon=True)
plt.savefig('cifar10-exp3.pdf', bbox_inches='tight')

### failure in model5 
original_acc = [0.7575, 0.654, 0.832, 0.54, 0.9085]
avg_drift = json.load(open('./data/cifar-exp3-drift.json', 'r'))
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(range(len(avg_drift)), avg_drift, color=colors[0],label='exp3-drift')
for i in range(5):
    ax.plot(range(len(avg_drift)), np.ones((len(avg_drift),))*original_acc[i], '--', color=colors[i+1],label='model'+str(i+1))
plt.axvline(x=1000,color='black',linestyle='--')
ax.annotate('failure in model5', xy=(1001, 0.9), xytext=(1300, 0.92),
            arrowprops=dict(facecolor='black', shrink=0.05), size=13
            )
plt.xlabel('queries', fontsize=15)
plt.ylabel('accuracy', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.set_ylim([0.5,1])
ax.legend(loc=4, frameon=True)
plt.savefig('cifar10-exp3-drift.pdf', bbox_inches='tight')
