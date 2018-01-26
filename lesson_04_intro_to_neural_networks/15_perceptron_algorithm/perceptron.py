import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    # print(X)
    return step_function((np.matmul(X, W) + b)[0])


# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptron_step(X, y, W, b, learn_rate=0.01):
    # Fill in code
    update_cnt = 0
    for i in range(len(X)):
        result = prediction(X[i], W, b)
        # print('before: %s - %s' % (W, b))
        if result < y[i]:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
            update_cnt += 1
        if result > y[i]:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b = b - learn_rate
            update_cnt += 1
        # print('after: %s - %s' % (W, b))
    print('update_cnt: %d' % update_cnt)

    return W, b, update_cnt

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptron_algorithm(X, y, learn_rate=0.01, num_epochs=100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    # print(W)
    b = np.random.rand(1)[0] + x_max
    # print(b)
    # These are the solution lines that get plotted below.
    boundary_lines = []
    neurons_updated = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b, nn_updated = perceptron_step(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
        neurons_updated.append(nn_updated)
    return boundary_lines, neurons_updated

data =  np.genfromtxt('data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

num_epochs=50
for learn_rate in np.arange(0.01, 0.10, 0.01):
    boundary_lines, neurons_updated = train_perceptron_algorithm(X, y, learn_rate, num_epochs)

    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, 1), ylim=(0, 1))
    ax.grid()

    blue = data[np.where(data[:, 2] == 0.), :-1][0]
    red = data[np.where(data[:, 2] == 1.), :-1][0]

    line, = ax.plot([], [], 'k-o', lw=2)
    blue_dots, = ax.plot([], [], 'bo')
    red_dots, = ax.plot([], [], 'ro')
    title_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    iteration_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    errors_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

    def init():
        """ initialize animation """
        line.set_data([], [])
        blue_dots.set_data(blue[:, 0], blue[:, 1])
        red_dots.set_data(red[:, 0], red[:, 1])
        iteration_text.set_text('')
        errors_text.set_text('')
        title_text.set_text('')
        return line, blue_dots, red_dots, iteration_text, errors_text, title_text

    def animate(i):
        """ animate """

        boundary_line = boundary_lines[i]
        x1 = 0
        y1 = x1 * boundary_line[0] + boundary_line[1]
        y1 = y1.item()
        x2 = 1
        y2 = x2 * boundary_line[0] + boundary_line[1]
        y2 = y2.item()
        line.set_data([x1, x2], [y1, y2])

        blue_dots.set_data(blue[:, 0], blue[:, 1])
        red_dots.set_data(red[:, 0], red[:, 1])

        iteration_text.set_text('iteration = %02d' % i)
        errors_text.set_text('updates = %02d' % neurons_updated[i])

        title_text.set_text("total epochs %d - learn_rate %0.2f" % (num_epochs, learn_rate))
        return line, blue_dots, red_dots, iteration_text, errors_text, title_text


    nn_animation = animation.FuncAnimation(fig, animate, frames=num_epochs,
                                  interval=200, blit=True, init_func=init)

    # save file
    filename = 'nn_%02d_%0.2f.mp4' % (num_epochs, learn_rate)
    nn_animation.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
