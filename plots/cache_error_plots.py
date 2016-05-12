import matplotlib
import matplotlib.pyplot as plt 
import json
import os


matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 10
nbins=8
# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
f = open('../results/cache_error.json','r')
res = json.load(f)

cache_miss_rate = res['cache_miss']
error_list = res['error']
fig,ax = plt.subplots()
plt.locator_params(nbins=nbins)
ax.scatter(cache_miss_rate, error_list,color="black", s=10)
#ax.plot([0.0,1.0],[error, error],'--', label="non-cache")
ax.set_xlabel('Cache Miss Rate')
ax.set_ylabel('Error')
#ax.set_ylim((0,ax.get_ylim()[1]))
# ax.legend()
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0.28,0.415)
fig.set_size_inches(3*1.5,1.0*1.5)
fig.savefig(fig_dir + '/' + 'cache_miss' +'.pdf',bbox_inches='tight')
