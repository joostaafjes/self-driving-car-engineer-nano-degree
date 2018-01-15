import matplotlib.pyplot as plt

plt.ion()
plt.plot([1.6, 2.7])
input("next")

plt.title("interactive test")
plt.xlabel("index")
input("next")

ax = plt.gca()
ax.plot([3.1, 2.2])
ax.draw()
