import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("pid_data_Sr89_E2.986_Kr86.npz", allow_pickle=True)
data2 = np.load("pid_data_Sr88_E2.986_Kr86.npz", allow_pickle=True)

E1 = data1["E_recoil"]
dE1 = data1["dE_recoil"]

E2 = data2["E_recoil"]
dE2 = data2["dE_recoil"]

plt.scatter(E1, dE1, s=2, label='89Sr')
plt.scatter(E2, dE2, s=2, marker='x', c='red', label='88Sr')
plt.xlabel("DSSD E (MeV)")
plt.ylabel("IC dE (MeV)")
plt.xlim(80,180)
plt.ylim(0,100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
