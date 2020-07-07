import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
import math


def display_multiple_boxplot(X, shape_given):
    # X dataset with the n variables for the boxplots
    # shape of the subplot

    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    for row, col in shape_given:
        fig, ax = plt.subplots(row, col, figsize = ((3.5*col),(4*row)))
        i, j = 0, 0

        for k in X.columns:
            if row > 1 and col > 1:
                if j < row:
                    ax[j][i].boxplot(x=X[k], showmeans=True, meanprops=meanprops, medianprops=medianprops)
                    ax[j][i].set_title('{}'.format(k))
                    ax[j][i].set_facecolor('whitesmoke')
                    ax[j][i].grid(True, c='white')

                    i += 1
                    if i == col:
                        (i, j) = (0, j+1)

            else:
                if i < max(row, col):
                    ax[i].boxplot(x=X[k], showmeans=True, meanprops=meanprops, medianprops=medianprops)
                    ax[i].set_title('{}'.format(k))
                    ax[i].set_facecolor('whitesmoke')
                    ax[i].grid(True, c='white')
                    i += 1
        return fig



def display_multiple_distplot(X, shape_given, title=None):
    # X dataset with the n variables for the boxplots
    # shape of the subplot

    for row, col in shape_given:
        fig, axes = plt.subplots(row, col, figsize = ((4*col),(5*row)))
        i, j = 0, 0

        for k in X.columns:
            if row > 1 and col > 1:
                if j < row:
                    sns.distplot(X[k], kde=False, ax=axes[j,i])
                    if title==None:
                        axes[j][i].set_title('{}'.format(k))

                    i += 1
                    if i == col:
                        (i, j) = (0, j+1)

            else:
                if i < max(row, col):
                    sns.distplot(X[k], kde=False, ax=axes[i])
                    if title==None:
                        axes[i].set_title('{}'.format(k))
                    i += 1
        return fig


def display_corr_matrix(X, title=None):
    corr = X.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    corr_matrix, ax = plt.subplots(figsize=(12, 10))
    # Generate a custom diverging colormap
    cmap = sns.color_palette("GnBu_d")
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, mask=mask,
                cmap=cmap, square=True, center=0, linewidths=.5, cbar_kws={"shrink": .7})
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22.5, horizontalalignment='right')

    if title is None:
        plt.title('Correlation Matrix', weight='bold')
    else:
        plt.title(title, weight='bold')
    plt.show()
    return corr_matrix


def display_multi_bivariate_boxplot(X_var, X, shape_given, significance=0.05):
    # X : dataframe, categorical variable (only 1) needs to be first in the index
    # shape_given : shape of the subplot (ex. [(3,2)], recommended to have only 2 in col

    sns.set_style("darkgrid")
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    for row, col in shape_given:
        fig, axes = plt.subplots(row, col, figsize=((7 * col), (8 * row)))
        i, j = 0, 0

        for Y_var in X.iloc[:, 1:].columns:
            linear_model = ols('{} ~C({})'.format(Y_var, X_var), data=X).fit()
            pvalue = linear_model.f_pvalue
            r_squared = linear_model.rsquared

            if pvalue < significance:
                title = '{} vs {} \n' \
                        'R-squared = {} and pvalue = {}'.format(
                    X_var, Y_var, round(r_squared, 3), round(pvalue, 3))

                if row > 1 and col > 1:
                    if j < row:
                        sns.boxplot(X[X_var], X[Y_var], showmeans=True, meanprops=meanprops,
                                        medianprops=medianprops, ax=axes[j,i])
                        axes[j, i].set_title(title)
                        i += 1
                        if i == col:
                            (i, j) = (0, j+1)

                else:
                    if i < max(row, col):
                        sns.boxplot(X[X_var], X[Y_var], showmeans=True,
                                        meanprops = meanprops, medianprops = medianprops, ax=axes[i])
                        axes[i].set_title(title)
                        i += 1
        return fig


def display_multiple_scatter(df, shape_given, significance):
    # df = dataframe source
    # significance = level of significance for the pvalue
    # shape_given : shape of the subplot (ex. [(3,2)], recommended to have only 2 in col


    for row, col in shape_given:
        fig, axes = plt.subplots(row, col, figsize=((7 * col), (10 * row)))
        i, j = 0, 0
        k = 0
        for X_var in df.columns:
            for y_var in df.iloc[:, k + 1:].columns:

                X = df[[X_var]]
                y = df[y_var]

                X = X.assign(intercept=[1] * X.shape[0])
                lr = sm.OLS(y, X).fit()

                r_squared = lr.rsquared
                pvalue = lr.f_pvalue

                if np.sqrt(r_squared) < 0.1:
                    effect = 'Small'
                elif np.sqrt(r_squared) >= 0.1 and np.sqrt(r_squared) < 0.3:
                    effect = 'Medium'
                else:
                    effect = 'Large'

                a, b = lr.params[X_var], lr.params['intercept']

                X = df[X_var]
                y = df[y_var]

                if pvalue < significance:
                    title = '{} vs {} \n' \
                            'R-squared = {} and pvalue = {} \n' \
                            '{} correlation'.format(X_var, y_var, round(r_squared, 3),round(pvalue, 3), effect)
                    if row > 1 and col > 1:
                        if j < row:
                            sns.scatterplot(X, y, ax=axes[j, i])
                            axes[j, i].plot(np.arange(min(X), max(X), 0.1),
                                            [a * x + b for x in np.arange(min(X), max(X), 0.1)], color='#f67575')
                            axes[j, i].set_title(title)
                            i += 1
                            if i == col:
                                (i, j) = (0, j + 1)
                    else:
                        if i < max(row, col):
                            sns.scatterplot(X, y, ax=axes[i])
                            axes[i].set_title(title)
                        i += 1
            k += 1

        return fig


