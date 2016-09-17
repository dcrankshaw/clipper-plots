import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#sns.set(style="white")
original_acc = [ 0.7292,  0.6615 ,  0.6802,  0.5603, 0.7761]
ensemble_acc = 0.7859
fig, ax = plt.subplots(figsize=(4.5,3))
four_agree_c = (0.8766, 0.74066)  ## the second is the proportion
four_agree_u = (0.5266, 0.25934)

five_agree_c = (0.9345, 0.48446)
five_agree_u = (0.6462, 0.51554)

bar_width = 0.3
opacity = 0.8

thin_bar = 0.08
color = sns.color_palette('bright', n_colors=4, desat=.5)

for i in range(len(original_acc)):
    if i == 0:
        rects0 = plt.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0], label='single model')
    else:
        plt.bar(thin_bar*(i+1), original_acc[i], thin_bar, alpha=opacity, color=color[0])

    
rects1 = plt.bar(0.5+bar_width, ensemble_acc, bar_width, alpha=opacity, color=color[1], label='ensemble')
rects2 = plt.bar(1.2+bar_width, four_agree_c[0], four_agree_c[1]*bar_width,  alpha=opacity, color=color[2], label='confident')
rects3 = plt.bar(1.12+bar_width*2, four_agree_u[0], four_agree_u[1]*bar_width,  alpha=opacity, color=color[3], label='unsure')
plt.bar(1.95+bar_width, five_agree_c[0], five_agree_c[1]*bar_width,  alpha=opacity, color=color[2])
plt.bar(2.1+bar_width, five_agree_u[0], five_agree_u[1]*bar_width,  alpha=opacity, color=color[3])


plt.ylabel('Top-1 Accuracy')
ax.set_ylim([0.5,1.1])

first_legend = plt.legend(handles=[rects0,rects1], loc=2)
axx = plt.gca().add_artist(first_legend)
second_legend = plt.legend(handles=[rects2,rects3], loc=1)


ind = np.array([0.3,1,1.7,2.4])
plt.xticks(ind, ('single model', 'ensemble', '4-agree', '5-agree'))

cnt = 0
for p in ax.patches:
    if cnt > 4:
        ax.annotate(str(float(p.get_height())), (p.get_x() * 1, p.get_height() * 1.01))
    elif cnt == 4:
    	ax.annotate(str(float(p.get_height())), (p.get_x() * 0.9, p.get_height() * 1.01))
    cnt += 1
plt.savefig('imagenet_ensemble_confidence_plot.pdf')

