import matplotlib.pyplot as plt
import numpy as np


filename = "../tracker_11_finetune.npy"
corr = np.load(filename, allow_pickle=True)
ad = []
re = []
te = []
conf = []
fig, axs = plt.subplots(1, 3)
for imfile, corr_list in corr.item().items():
    
    for corr_dict in corr_list:
        if corr_dict["RE"] < 25000:
            ad.append(corr_dict["ADD"])
            re.append(corr_dict["RE"])
            te.append(corr_dict["TE"])
            conf.append(corr_dict["Conf"])
    
axs[0].scatter(ad, conf)
axs[0].set_title('ADD')

axs[1].scatter(re, conf)
axs[1].set_title('RE')

axs[2].scatter(te, conf)
axs[2].set_title('TE')
print(len(ad))
plt.show()