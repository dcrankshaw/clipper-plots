import matplotlib.pyplot as plt 
import json


f = open('cache_error.json','r')
res = json.load(f)

cache_miss_rate = res['cache_miss']
error_list = res['error']
fig,ax = plt.subplots()
ax.scatter(cache_miss_rate, error_list)
#ax.plot([0.0,1.0],[error, error],'--', label="non-cache")
ax.set_xlabel('cache miss rate')
ax.set_ylabel('error rate')
#ax.set_ylim((0,ax.get_ylim()[1]))
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.1
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()