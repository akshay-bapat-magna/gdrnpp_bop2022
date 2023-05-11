import numpy as np
import matplotlib.pyplot as plt

# Read data from log
log_file = "bayesian_opt_log.txt"
run_date = "2023-04-04"
lr = []
bs = []
mmt = []
add = []
with open(log_file, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines) and run_date not in lines[i]:
        i += 1
    
    i += 2
    while i < len(lines):
        data = lines[i].split(', ')
        lr.append(-np.log10(float(data[0].split()[-1])))
        bs.append(int(data[1].split()[-1]))
        mmt.append(float(data[2].split()[1]))
        add.append(float(data[2].split()[-1].split('\n')[0]))
        i += 1

# Plot data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Batch Size')
ax.set_zlabel('Momentum')
assert len(lr) == len(bs) == len(mmt) == len(add)
# for i in range(len(lr)):
#     if i < 40:
#         ax.scatter(lr[i], bs[i], mmt[i], c=add[:40], marker='o', cmap='winter')
#     else:
#         ax.scatter(lr[i], bs[i], mmt[i], c=add[40:], marker='x', cmap='winter')
    # plt.pause(1)

exp = 40
plot = ax.scatter(lr[:exp], bs[:exp], mmt[:exp], c=add[:exp], marker='o', cmap='winter')
plot = ax.scatter(lr[exp:], bs[exp:], mmt[exp:], c=add[exp:], marker='x', cmap='winter')
opt = np.argmax(add)
plot = ax.scatter(lr[opt], bs[opt], mmt[opt], c=add[opt], marker='^', cmap='winter', linewidths=3)
fig.colorbar(plot, label="ADD5 Recall")
plt.show()