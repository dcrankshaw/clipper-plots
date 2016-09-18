import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style("white")
sns.set_context("paper", font_scale=1.0)

color = sns.color_palette('bright', n_colors=4, desat=.5)

fig, (ax_cifar, ax_imgnet) = plt.subplots(nrows=2, sharex=False, figsize=(5,6), gridspec_kw = {'height_ratios':[2, 2]})


original_acc = [ 0.7292,  0.6615 ,  0.6802,  0.5603, 0.7761]
ensemble_acc = 0.7859
four_agree_c = (0.8766, 0.74066)  ## the second is the proportion
four_agree_u = (0.5266, 0.25934)

five_agree_c = (0.9345, 0.48446)
five_agree_u = (0.6462, 0.51554)
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
rects3 = ax_imgnet.bar(1.12+bar_width*2, four_agree_u[0], four_agree_u[1]*bar_width,  alpha=opacity, color=color[3], label='unsure')
ax_imgnet.bar(1.95+bar_width, five_agree_c[0], five_agree_c[1]*bar_width,  alpha=opacity, color=color[2])
ax_imgnet.bar(2.1+bar_width, five_agree_u[0], five_agree_u[1]*bar_width,  alpha=opacity, color=color[3])


ax_imgnet.set_ylabel('Top-1 Accuracy', fontsize=13)
ax_imgnet.set_ylim([0.5,1.1])
ax_imgnet.set_title('Imagenet', fontsize=11)

first_legend = ax_imgnet.legend(handles=[rects0,rects1], loc=2, fontsize=9)
ax_imgnet.add_artist(first_legend)
second_legend = ax_imgnet.legend(handles=[rects2,rects3], loc=1, fontsize=9)

ind = np.array([0.3,1,1.7,2.4])
ax_imgnet.set_xticks(ind)
ax_imgnet.set_xticklabels(['single model', 'ensemble', '4-agree', '5-agree'])
ax_imgnet.tick_params(axis='both', which='major', labelsize=10)
ax_imgnet.tick_params(axis='both', which='minor', labelsize=10)

cnt = 0
for p in ax_imgnet.patches:
    if cnt > 4:
        ax_imgnet.annotate(str(float(p.get_height())), (p.get_x() * 1, p.get_height() * 1.01))
    elif cnt == 4:
    	ax_imgnet.annotate(str(float(p.get_height())), (p.get_x() * 0.9, p.get_height() * 1.01))
    cnt += 1



###### cifar10 
four_agree_c = (0.9390, 0.8035)  ## the second is the proportion
four_agree_u = (0.8193, 0.1965)

five_agree_c = (0.9765, 0.405)
five_agree_u = (0.8740, 0.595)
original_acc = [0.7575, 0.654, 0.832, 0.54, 0.9085]
ensemble_acc = 0.9155

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


ax_cifar.set_ylabel('Top 1 Accuracy', fontsize=13)
ax_cifar.set_ylim([0.5,1.16])
ax_cifar.set_title('Cifar10', fontsize=11)

first_legend = ax_cifar.legend(handles=[rects0,rects1], loc=2, fontsize=9)
ax_cifar.add_artist(first_legend)
second_legend = ax_cifar.legend(handles=[rects2,rects3], loc=1, fontsize=9)


ind = np.array([0.3,1,1.7,2.4])
ax_cifar.set_xticks(ind)
ax_cifar.set_xticklabels(['single model', 'ensemble', '4-agree', '5-agree'])
ax_cifar.tick_params(axis='both', which='major', labelsize=10)
ax_cifar.tick_params(axis='both', which='minor', labelsize=10)

cnt = 0
for p in ax_cifar.patches:
    if cnt >= 4:
        ax_cifar.annotate(str(float(p.get_height())), (p.get_x() * 1.01, p.get_height() * 1.01))
    cnt += 1

plt.savefig('robust-prediction.pdf', bbox_inches='tight')


