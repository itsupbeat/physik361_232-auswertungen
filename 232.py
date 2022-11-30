import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif'
})


colors = ['#046865', '#111d41', '#7D4E57', '#647582', '#d13c17']


def errorbar(x, x_err, y, y_err, names, colors, j=1, beschriftung=['', '', ''], dateiname=f'out.pdf'):
    """
    Function to create an errorbar graph for multiple y-Values all having same x value-array.
    :param x:
    :param x_err:
    :param y:
    :param y_err:
    :param names:
    :param colors:
    :param j: number of y-value-arrays
    :param beschriftung:
    :param dateiname:
    :return:
    """
    fig, ax = plt.subplots()

    for n in range(j):
        print(n)
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=names[n], fmt=',', color=colors[4],
                    elinewidth=0.1, ecolor=colors[3])
    data_range = np.linspace(0, 1500, np.size(x) * 5)
    y_fit = lin_model(data_range, 0.6, 5)
    y_fit_2 = lin_model(data_range, 0.1136, 4.413)
    ax.plot(data_range, y_fit, label=r'$\mu_\textrm{\tiny{max}}$', color=colors[1], linewidth=1)
    ax.plot(data_range, y_fit_2, label=r'$\mu_\textrm{\tiny{A}}$', color=colors[2], linewidth=1)
    ax.grid(True)
    ax.set_title(beschriftung[0])
    ax.set_xlabel(beschriftung[1])
    ax.set_ylabel(beschriftung[2])
    ax.legend()

    fig.savefig(dateiname)
    plt.show()


def xy_data(x_data, y_data, x_err=0, y_err=0):
    """
    Function to make code more clear. Builds the XYContainer used by kafe2
    :param x_data: x-data for fit
    :param y_data: y-data for fit
    :param x_err: error of x-data for fit
    :param y_err: error of y-data for fit
    :return:
    """
    data = XYContainer(x_data=x_data, y_data=y_data)
    data.add_error(axis='x', err_val=x_err)
    data.add_error(axis='y', err_val=y_err)
    return data


def lin_model(x, a, b):
    """
    Fit model for 370.b
    :param x: the inverse wavelength squared (not just wavelength)
    :param a: constant
    :param b: constant
    :return:
    """
    return x * a + b


def lin_fit(x, y, x_err, y_err, title="", x_name="", y_name="", filename=f"lin-fit", colors=np.zeros(5)):
    """
    Function for linear fit
    :param colors: colors of graph elements
    :param x: values x axis
    :param y: values y axis
    :param x_err: error x axis
    :param y_err: error y axis
    :param title: string used as header
    :param x_name: string used as label for x axis
    :param y_name: string used as label for y axis
    :param filename: name your exported file
    :return:
    """
    data = xy_data(x, y, x_err, y_err)

    fit = Fit(data=data, model_function=lin_model)
    results = fit.do_fit()
    fit.report()

    a = results['parameter_values']['a']
    b = results['parameter_values']['b']

    edges = np.ptp(x)*.05
    data_range = np.linspace(np.amin(x)-edges, np.amax(x)+edges, np.size(x) * 5)
    y_fit = lin_model(data_range, a, b)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, fmt='H', xerr=x_err, yerr=y_err, label='Datenpunkte', color=colors[0],
                capsize=1, ms=3, elinewidth=1)
    ax.plot(data_range, y_fit, label='Fit', color=colors[1], linewidth=1)

    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    ax.legend()

    fig.savefig(f'{filename}.pdf')

    plt.show()

