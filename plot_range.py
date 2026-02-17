import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./VICAR_outputs/Kr86_helium_230psi.txt", sep="\t", names=["e","r"])
# plt.scatter(df['e'],df['r'])
df.plot.scatter(x='e',y='r')
plt.show()