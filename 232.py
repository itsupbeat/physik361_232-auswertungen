import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif'
})


colors = ['#046865', '#111d41', '#7D4E57', '#647582', '#d13c17', '#046865', '#111d41', '#7D4E57', '#647582', '#d13c17']
# lin_fit_colors = []


def errorbar(x, y, x_err, y_err, names, colors, j=1, beschriftung=['', '', ''], filename=f'out.pdf'):
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
    if j > 1:
        for n in range(j):
            print(n)
            ax.errorbar(x[n], y[n], xerr=x_err[n], yerr=y_err[n], label=names[n], fmt='H', color=colors[n],
                        elinewidth=1, ecolor=colors[n])
    elif j == 1:
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=names, fmt='H', color=colors[0],
                    elinewidth=1, ecolor=colors[0])
    # data_range = np.linspace(0, 1500, np.size(x) * 5)
    # y_fit = lin_model(data_range, 0.6, 5)
    # y_fit_2 = lin_model(data_range, 0.1136, 4.413)
    # ax.plot(data_range, y_fit, label=r'$\mu_\textrm{\tiny{max}}$', color=colors[1], linewidth=1)
    # ax.plot(data_range, y_fit_2, label=r'$\mu_\textrm{\tiny{A}}$', color=colors[2], linewidth=1)
    ax.grid(True)
    ax.set_title(beschriftung[0])
    ax.set_xlabel(beschriftung[1])
    ax.set_ylabel(beschriftung[2])
    ax.legend()

    fig.savefig(f'{filename}.pdf')
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


def lin_model(x, a=140, b=0):
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


def lin_fit_mult(x, y, x_err, y_err, title="", x_name="", y_name="", counter=1, names=[0, 0, 0, 0], filename=f"lin-fit", colors=np.zeros(10)):
    """
    Function for linear fit
    :param names:
    :param colors: colors of graph elements
    :param x: values x axis
    :param y: values y axis
    :param x_err: error x axis
    :param y_err: error y axis
    :param title: string used as header
    :param x_name: string used as label for x axis
    :param y_name: string used as label for y axis
    :param counter
    :param filename: name your exported file
    :return:
    """
    fig, ax = plt.subplots()
    for n in range(counter):
        data = xy_data(x, y[n], x_err, y_err[n])
        print('###\n' + names[n] + '\n###')
        fit = Fit(data=data, model_function=lin_model)
        results = fit.do_fit()
        fit.report()

        a = results['parameter_values']['a']
        b = results['parameter_values']['b']

        edges = np.ptp(x)*.05
        data_range = np.linspace(np.amin(x)-edges, np.amax(x)+edges, np.size(x) * 5)
        y_fit = lin_model(data_range, a, b)

        ax.errorbar(x, y[n], fmt='H', xerr=x_err, yerr=y_err[n], label=names[n], #color=colors[2*n],
                    capsize=1, ms=3, elinewidth=1)
        ax.plot(data_range, y_fit, # color=colors[2*n+1],
                linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    ax.legend()

    fig.savefig(f'{filename}.pdf')

    plt.show()


#################
# Aufgabe 232.n #
#################

m_metalle = np.loadtxt(f'232-m-metalle.csv', skiprows=1)
m_temp = m_metalle[:, 0]
m_temp_err = m_metalle[:, 1]
m_konst = m_metalle[:, 2]
m_konst_err = m_metalle[:, 3]
m_plat = m_metalle[:, 4]
m_plat_err = m_metalle[:, 5]
m_kohl = m_metalle[:, 6]
m_kohl_err = m_metalle[:, 7]

m_metalle_y = np.array([m_konst, m_plat, m_kohl])
m_metalle_y_err = np.array([m_konst_err, m_plat_err, m_kohl_err])
m_metalle_names = [r'Konstantanwiderstand', r'Platinwiderstand', r'Kohleschichtwiderstand']

# lin_fit_mult(m_temp, m_metalle_y, m_temp_err, m_metalle_y_err, r'Temperaturabhängigkeit metallischer Leiter',
#         r'$T\ [^\circ \textrm{C}]$', r'$R\ [\Omega]$', 3, m_metalle_names, f'm-metalle')#, lin_fit_colors)

m_halbleiter = np.loadtxt(f'232-m-hallbleiter.csv', skiprows=1)
m_t_invers = m_halbleiter[:, 0]
m_t_invers_err = m_halbleiter[:, 1]
m_ln_ntp = m_halbleiter[:, 2]
m_ln_ntp_err = m_halbleiter[:, 3]

# lin_fit(m_t_invers, m_ln_ntp, m_t_invers_err, m_ln_ntp_err, r'Temperaturabh\"angigkeit des NTP-Widerstands',
#         r'$1/T\ [1/\textrm{K}]$', r'$\ln (R)\ [\ ]$', f'n_nichtleiter', colors)

m_ptc = np.loadtxt(f'232-m-ptc.csv', skiprows=1)
m_ln_ptc = m_ptc[:, 2]
m_ln_ptc_err = m_ptc[:, 3]

# errorbar(m_temp, m_ln_ptc, m_temp_err, m_ln_ptc_err, [r'PTC-Widerstand'], colors, 1,
#          [r'Temperaturabh\"angigkeit des PTC-Widerstands', r'$1/T\ [1/\textrm{K}]$', r'$\ln (R)\ [\ ]$'], f'n_ptc')

###
# Aufgabe 232.a
##

a_data = np.loadtxt(f'232_a.csv', skiprows=1)
print(a_data)

# lin_fit(a_data[:, 8], a_data[:, 10], a_data[:, 9], a_data[:, 11],
#         r'Widerstandsbestimmung durch Spannungs- und Stromabh\"angigkeit', r'$I\ [\textrm{A}]$', r'$U\ [\textrm{V}]$',
#         f'232_a', colors)

##
# Aufgabe 232.f
#####

f_inf = np.loadtxt(f'232_f_inf.txt', skiprows=1)
f_inf_r = f_inf[:, 4]
f_inf_r_err = f_inf[:, 5]
f_inf_u = f_inf[:, 6]
f_inf_u_err = f_inf[:, 7]

f_zwan = np.loadtxt(f'232_f_20.txt', skiprows=1)
f_zwan_r = f_zwan[:, 4]
f_zwan_r_err = f_zwan[:, 5]
f_zwan_u = f_zwan[:, 6]
f_zwan_u_err = f_zwan[:, 7]

f_funf = np.loadtxt(f'232_f_50.txt', skiprows=1)
f_funf_r = f_funf[:, 4]
f_funf_r_err = f_funf[:, 5]
f_funf_u = f_funf[:, 6]
f_funf_u_err = f_funf[:, 7]

f_x = np.array([f_inf_r, f_zwan_r, f_funf_r])
f_x_err = np.array([f_inf_r_err, f_zwan_r_err, f_funf_r_err])
f_y = np.array([f_inf_u, f_zwan_u, f_funf_u])
f_y_err = np.array([f_inf_u_err, f_zwan_u_err, f_funf_u_err])

errorbar(f_x, f_y, f_x_err, f_y_err, [r'$R_\textrm{\tiny{L}}=\infty\ \Omega$', r'$R_\textrm{\tiny{L}}=20\ \Omega$',
                                      r'$R_\textrm{\tiny{L}}=50\ \Omega$'], colors, j=3,
         beschriftung=[r'Spannungsabfall bei verschiedenen $R_\textrm{tiny{L}}$', r'$R\ [\Omega]$',
                       r'$U\ [\textrm{V}]$'], filename=f'f')


