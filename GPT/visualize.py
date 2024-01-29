import matplotlib.pyplot as plt
import statistics

def visualize_ent_scatter(num_tokens: list, num_labels: list, entity: str):
    plt.scatter(num_tokens, num_labels, alpha=0.5)
    plt.title("Relationship between the number of tokens and the number of " +
              entity + " entities")
    plt.xlabel('Number of tokens')
    plt.ylabel("Number of " + entity + " entities")
    plt.show()

def hist_distrib(axis, num_per_note:list[int], name:str, color="blue"):

    axis.hist(num_per_note, bins=20, color=color, alpha=0.7, label=name)
    axis.axvline(statistics.mean(num_per_note), color='red', linestyle='dashed', linewidth=2, label='Mean')
    axis.axvline(statistics.mean(num_per_note) + statistics.stdev(num_per_note), color='purple', linestyle='dashed', linewidth=2, label='Mean + Standard deviation')
    axis.axvline(statistics.mean(num_per_note) - statistics.stdev(num_per_note), color='purple', linestyle='dashed', linewidth=2, label='Mean - Standard deviation')
    axis.set_xlabel('Number of ' + name)
    axis.set_ylabel('Frequency')
    axis.set_title("Number of " + name +" in the notes")
    axis.legend()

def visualize_distrib_outliers(num_tokens: list, num_labels: list, entity: str):
    fig, axis = plt.subplots(1, 2, figsize=(12, 4))

    hist_distrib(axis[0], num_tokens, "tokens")
    hist_distrib(axis[1], num_labels, entity + " entities", color="green")

    plt.tight_layout()
    plt.show()


def box_plot(values:list[int], name:str):
    plt.boxplot(values)

    plt.title('Box Plot')
    plt.ylabel(name)

    plt.show()