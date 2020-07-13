import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pandas.plotting import parallel_coordinates
from matplotlib.collections import LineCollection

import statsmodels.api as sm
from statsmodels.formula.api import ols
import math

from scipy.cluster.hierarchy import dendrogram



####################
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    sns.set_style("dark")
    for d1, d2 in axis_ranks:
        if d2 < n_comp:


            fig, ax = plt.subplots(figsize=(12, 12))
            plt.title("Correlation Circle F{} et F{}".format(d1 + 1, d2 + 1))

            # limits
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1*1.05, 1*1.05, -1*1.05, 1*1.05
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])
                plt.title("- Zoom - Correlation Circle (F{} et F{})".format(d1 + 1, d2 + 1))

            # arrow display
            # if more than 30 arrows, do not display end of the arrow
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', width=0.003, scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # variable display
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='8', ha='center', va='center', rotation=label_rotation,
                                 color="black")

            # circle display
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black')
            plt.gca().add_artist(circle)

            # define limits
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # display middle lines
            plt.plot([-1, 1], [0, 0], color='silver', ls='-', linewidth=1)
            plt.plot([0, 0], [-1, 1], color='silver', ls='-', linewidth=1)

            # axes names, with explained variance %
            plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.show(block=False)
            return fig

############

def display_factorial_planes(X_projected, n_comp,
                             pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, lims=None):


    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            sns.set_style("darkgrid")

            fig = plt.figure(figsize=(12, 12))
            x_axis = X_projected[:, d1]
            y_axis = X_projected[:, d2]

            # graph limits
            if lims is None:
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                xmin, ymin = -boundary, -boundary
                xmax, ymax = boundary, boundary
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
            else:
                xmin, xmax, ymin, ymax = lims
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)

            # display points
            if illustrative_var is None:
                sns.scatterplot(x_axis, y_axis, alpha=alpha)
            else:
                sns.scatterplot(x_axis, y_axis, hue=illustrative_var, alpha=alpha, palette = 'Set3')

            # display centroid only if there is no labels to display
                if labels is None:
                    centroids_all = []
                    for cluster_name in sorted(illustrative_var.unique()):
                        index_proj = illustrative_var.loc[illustrative_var == cluster_name].index
                        x_centroid, y_centroid = (np.mean(x_axis[index_proj]), np.mean(y_axis[index_proj]))
                        centroids_all.append([x_centroid, y_centroid])

                    plt.scatter([row[0] for row in centroids_all], [row[1] for row in centroids_all], s=85, alpha=0.5, marker='o', c='black', label='Centroids')

                    for i, (x, y) in enumerate(centroids_all):
                        plt.text(x, y + 0.05, sorted(illustrative_var.unique())[i], fontsize='10', weight='bold',
                                 ha='center', va='bottom')

            # display labels of the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y + 0.05, labels[i], fontsize='8', ha='center', va='bottom')

            # display middle lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # axes names with value
            plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.legend()
            plt.title("Projection on F{} and F{}".format(d1 + 1, d2 + 1))

            plt.show(block=False)
            return fig


#############

def display_factorial_planes_2(X_projected, n_comp,
                             pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, illustrative_var_2=None, lims=None):


    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            sns.set_style("darkgrid")

            fig = plt.figure(figsize=(12, 12))
            x_axis = X_projected[:, d1]
            y_axis = X_projected[:, d2]

            # graph limits
            if lims is None:
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                xmin, ymin = -boundary, -boundary
                xmax, ymax = boundary, boundary
                plt.xlim([xmin, xmax])
                plt.ylim([ymin, ymax])
            else:
                xmin, xmax, ymin, ymax = lims
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)

            # display points
            if illustrative_var is None:
                sns.scatterplot(x_axis, y_axis, alpha=alpha)
            else:
                sns.scatterplot(x_axis, y_axis, hue=illustrative_var_2, alpha=alpha, palette = 'Set3')

            # display centroid only if there is no labels to display
                if labels is None:
                    centroids_all = []

                    for cluster_name in sorted(illustrative_var.unique()):
                        index_proj = illustrative_var.loc[illustrative_var == cluster_name].index
                        x_centroid, y_centroid = (np.mean(x_axis[index_proj]), np.mean(y_axis[index_proj]))
                        centroids_all.append([x_centroid, y_centroid])

                    plt.scatter([row[0] for row in centroids_all], [row[1] for row in centroids_all], s=85, alpha=0.5, marker='o', c='black', label='Centroids')

                    for i, (x, y) in enumerate(centroids_all):
                        plt.text(x, y + 0.05, sorted(illustrative_var.unique())[i], fontsize='10', weight='bold',
                                 ha='center', va='bottom')

            # display labels of the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y + 0.05, labels[i], fontsize='8', ha='center', va='bottom')

            # display middle lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # axes names with value
            plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.legend()
            plt.title("Projection on F{} and F{}".format(d1 + 1, d2 + 1))

            plt.show(block=False)
            return fig


###############

def plot_dendrogram(Z, names, orientation='left', display=None):
    sns.set_style("dark")

    if orientation in ['left', 'right']:
        fig = plt.figure(figsize=(10,25))
    else:
        fig = plt.figure(figsize=(25,10))

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = orientation,
    )

    if display is None:
        plt.show(fig)

    else:
        return fig


#############
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_ * 100
    n_comp = len(scree)

    bs = 1 / np.arange(n_comp, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]

    fig = plt.figure()
    plt.bar(np.arange(n_comp)+1, scree)
    plt.plot(np.arange(n_comp)+1, scree.cumsum(),c="black",marker='o')
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree plot - Explained variance vs # of factors")
    plt.xticks(np.arange(n_comp)+1)
    plt.show()
    return fig

###
palette = sns.color_palette("bright", 10)

####
def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''

    return (colour[0], colour[1], colour[2], alpha)

#########
def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster == i])

    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):
        plt.subplot(num_clusters, 1, i + 1)
        for j, c in enumerate(cluster_points):
            if i != j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j], 0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i], 0.5)])

        # Stagger the axes
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)

    plt.show()
    return fig

#######
def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    plt.show()
    return fig