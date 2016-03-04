import json

f = open('~/result/cold_start.json','r')
res = json.load(f)

## plot ###
fig,ax = plt.subplots()
iters = res['num_points']
ax.plot(iters, res['oracle'], label='oracle')
ax.plot(iters, res['l2'], label = 'svm-l2')
ax.plot(iters, res['l1'], label = 'svm-l1')
ax.plot(iters, res['independent'], label = 'non-sharing')
ax.set_xlabel('Size of the new task')
ax.set_ylabel('Error rate')
ax.set_title('Cold Start')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()
