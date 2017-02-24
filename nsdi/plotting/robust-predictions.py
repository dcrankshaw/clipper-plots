from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import os
#sns.set_style("white")
sns.set_context("paper", font_scale=1.0)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# color = sns.color_palette('bright', n_colors=4, desat=.5)
color = sns.cubehelix_palette(4, start=.75, rot=-.75)

fig, (ax_cifar, ax_imgnet) = plt.subplots(nrows=2, sharex=False, figsize=(5,3), gridspec_kw = {'height_ratios':[2, 2]})


#original_acc = [ 0.7292,  0.6615 ,  0.6802,  0.5603, 0.7761]
original_acc = [0.0618,  0.0902,  0.1152, 0.1305, 0.2061]
original_acc.reverse() ### put the best model in the end

#ensemble_acc = 0.7859
ensemble_acc = 0.0586
four_agree_c = (0.0469,  0.95676)  ## the second is the proportion
four_agree_u = (0.3182, 0.04324)

five_agree_c = (0.0327, 0.84366)
five_agree_u = (0.1983, 0.15634)
bar_width = 0.3
opacity = 1
thin_bar = 0.08

for i in range(len(original_acc)):
    if i == 0:
        rects0 = ax_imgnet.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0], label='single model')
    else:
        ax_imgnet.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0])

    
rects1 = ax_imgnet.bar(0.5+bar_width, ensemble_acc, bar_width, alpha=opacity, color=color[1], label='ensemble')
rects2 = ax_imgnet.bar(1.2+bar_width, four_agree_c[0], four_agree_c[1]*bar_width,  alpha=opacity, color=color[2], label='confident')
rects3 = ax_imgnet.bar(1.18+bar_width*2, four_agree_u[0], four_agree_u[1]*bar_width,  alpha=opacity, color=color[3], label='unsure')
ax_imgnet.bar(1.95+bar_width, five_agree_c[0], five_agree_c[1]*bar_width,  alpha=opacity, color=color[2])
ax_imgnet.bar(2.2+bar_width, five_agree_u[0], five_agree_u[1]*bar_width,  alpha=opacity, color=color[3])


ax_imgnet.set_ylabel('Top-5 Error Rate', fontsize=13)
ax_imgnet.set_ylim([0.0,0.55])
ax_imgnet.set_title('ImageNet', fontsize=11)

first_legend = ax_imgnet.legend(handles=[rects0,rects1], loc=2, ncol=2, fontsize=9)
ax_imgnet.add_artist(first_legend)
second_legend = ax_imgnet.legend(handles=[rects2,rects3], loc=1, ncol=1, fontsize=9)

ind = np.array([0.3,1,1.7,2.4])
ax_imgnet.set_xticks(ind)
ax_imgnet.set_xticklabels(['single model', 'ensemble', '4-agree', '5-agree'])
ax_imgnet.tick_params(axis='both', which='major', labelsize=10)
ax_imgnet.tick_params(axis='both', which='minor', labelsize=10)
ax_imgnet.locator_params(nbins=4, axis="y")

cnt = 0
for p in ax_imgnet.patches:
    if cnt >= 4:
        ax_imgnet.annotate(str(float(p.get_height())), (p.get_x() * 0.98, p.get_height() * 1.15))
    cnt += 1



###### cifar10 
four_agree_c = (0.0610, 0.8035)  ## the second is the proportion
four_agree_u = (0.1807, 0.1965)

five_agree_c = (0.0235, 0.405)
five_agree_u = (0.1260, 0.595)
original_acc = [0.46, 0.3461, 0.2425, 0.168,0.0915]
ensemble_acc = 0.0845

for i in range(len(original_acc)):
    if i == 0:
        rects0 = ax_cifar.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0], label='single model')
    else:
        ax_cifar.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0])

    
rects1 = ax_cifar.bar(0.5+bar_width, ensemble_acc, bar_width, alpha=opacity, color=color[1], label='ensemble')
rects2 = ax_cifar.bar(1.2+bar_width, four_agree_c[0], four_agree_c[1]*bar_width,  alpha=opacity, color=color[2], label='confident')
rects3 = ax_cifar.bar(1.14+bar_width*2, four_agree_u[0], four_agree_u[1]*bar_width,  alpha=opacity, color=color[3], label='unsure')
ax_cifar.bar(1.95+bar_width, five_agree_c[0], five_agree_c[1]*bar_width,  alpha=opacity, color=color[2])
ax_cifar.bar(2.07+bar_width, five_agree_u[0], five_agree_u[1]*bar_width,  alpha=opacity, color=color[3])


ax_cifar.set_ylabel('Top 1 Error Rate', fontsize=13)
ax_cifar.set_ylim([0.0, 0.6])
ax_cifar.set_title('CIFAR-10', fontsize=11)

first_legend = ax_cifar.legend(handles=[rects0,rects1], loc=2, ncol=2, fontsize=9)
ax_cifar.add_artist(first_legend)
second_legend = ax_cifar.legend(handles=[rects2,rects3], loc=1, fontsize=9)


ind = np.array([0.3,1,1.7,2.4])
ax_cifar.set_xticks(ind)
ax_cifar.set_xticklabels(['single model', 'ensemble', '4-agree', '5-agree'])
ax_cifar.tick_params(axis='both', which='major', labelsize=10)
ax_cifar.tick_params(axis='both', which='minor', labelsize=10)
ax_cifar.locator_params(nbins=4, axis="y")

cnt = 0
for p in ax_cifar.patches:
    if cnt == 8:
        ax_cifar.annotate(str(float(p.get_height())), (p.get_x() * 0.91, p.get_height() * 1.15))
    elif cnt >= 4 :
        ax_cifar.annotate(str(float(p.get_height())), (p.get_x() * 0.995, p.get_height() * 1.15))
    cnt += 1
fig.subplots_adjust(hspace=0.5)
fname = os.path.join(utils.NSDI_FIG_DIR, "robust-predictions.pdf")
plt.savefig(fname, bbox_inches='tight')
print(fname)


