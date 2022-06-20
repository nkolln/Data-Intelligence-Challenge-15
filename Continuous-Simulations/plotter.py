import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, config_list, count_test = None, simulationnr_stop = None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('lr= ' + config_list[0] + " optimizer " + config_list[1] + " criterion " + config_list[2])
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if simulationnr_stop != None and len(scores) == simulationnr_stop:
        plt.savefig("plots_discreteq/test_fig_" + str(count_test) + ".png")
