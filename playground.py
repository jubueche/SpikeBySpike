import numpy as np
import matplotlib.pyplot as plt
import warnings


r_rein = np.load("scaled_rewards_adaptive_learning_reinforcement10_0.dat",allow_pickle=True)
r_n_rein = np.load("scaled_rewards_adaptive_learning_no_reinforcement10_0.dat",allow_pickle=True)

title_font_size = 6
axis_font_size = 5
ticks_font_size = 4
linewidth = 0.5

C = np.load("DYNAPS/Resources/DYNAPS/Cs.dat", allow_pickle=True)[-1,:,:]
#C = np.load("DYNAPS/Resources/Simulation/Cs.dat", allow_pickle=True)[-1,:,:]
C_disc = np.load("DYNAPS/Resources/DYNAPS/Cs_discrete.dat", allow_pickle=True)[-1,:,:]


plt.figure(figsize=(3.94,1.57))
plt.subplot(121)
plt.matshow(C, fignum=False)
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.matshow(C_disc, fignum=False)
plt.xlabel("Neuron id", fontname="Times New Roman" ,fontsize=axis_font_size)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
plt.savefig("Latex/figures/weights.pdf", format='pdf', dpi=1200)
plt.show()


"""plt.figure(figsize=(6.0,1.18))
plt.title('Reward over time with and without RL based learning rate adaptation', fontname="Times New Roman" ,fontsize=title_font_size)
plt.plot(-r_rein, linewidth=linewidth, label="With RL")
plt.plot(-r_n_rein, linewidth = linewidth, label="No RL")
ax = plt.gca()
ax.set_xlabel('Iteration', fontname="Times New Roman" ,fontsize=axis_font_size)
ax.set_ylabel('Reward', fontname="Times New Roman" ,fontsize=axis_font_size)
ax.tick_params(axis='y', labelsize=ticks_font_size)
ax.tick_params(axis='x', labelsize=ticks_font_size)
L = ax.legend(prop={'size': 5})
plt.setp(L.texts, family='Times New Roman',fontsize=5)
plt.tight_layout()
plt.savefig("DYNAPS/Resources/Simulation/rl_based_lr_adaptation.eps", format='eps')
plt.show()"""

