'''
                    Copyright Oliver Kowalke 2020.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Based on the book 'Way Beyond Monochrome' by Lambrecht, Woodhouse
'''

import argparse
import ast
import configparser
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from prettytable import PrettyTable
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import newton


z2_reflection_density = 1.88
z3_reflection_density = 1.60
z5_reflection_density = 0.76
z7_reflection_density = 0.20
z8_reflection_density = 0.10


def show(x, y):
    fig, ax = plt.subplots()
    ax.grid(which='both')
    line = np.linspace(x[0], x[-1], 100)
    ax.plot(line, y(line), linewidth=1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.show()


def plot_filtration_chart(file_p, paper, enlarger, developer,
        IDmin, IDmax,
        textual_log_exposure_ranges,
        filter_yellow, filter_magenta):
    fig, ax = plt.subplots()
    ax.set_title('{} {}\n{} ({})\n{} f/{} {}s IDmin = {} IDmax = {}'.format(
        paper['name'], paper['surface'],
        enlarger['name'], enlarger['lamp'],
        developer['developer'], enlarger['f'], enlarger['t'],
        IDmin, IDmax,
        fontsize=5))
    ax.grid(which='both')
    line = np.linspace(textual_log_exposure_ranges[0], textual_log_exposure_ranges[-1], 500)
    ax.plot(line, filter_yellow(line), linewidth=1.0, color='yellow')
    ax.plot(line, filter_magenta(line), linewidth=1.0, color='magenta')
    ax.vlines(0.35, 0, 130, linestyles='dotted')
    ax.vlines(0.50, 0, 130, linestyles='dotted')
    ax.vlines(0.65, 0, 130, linestyles='dotted')
    ax.vlines(0.80, 0, 130, linestyles='dotted')
    ax.vlines(0.95, 0, 130, linestyles='dotted')
    ax.vlines(1.15, 0, 130, linestyles='dotted')
    ax.vlines(1.40, 0, 130, linestyles='dotted')
    ax.vlines(1.70, 0, 130, linestyles='dotted')
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('log exposure range')
    ax.set_ylabel('Durst filter density (130 max)')
    file_p = file_p.with_suffix('.png')
    plt.savefig(str(file_p), dpi=300)


def plot_filtration_tbl(file_p, paper, enlarger, filter_yellow, filter_magenta):
    print('{} {}'.format(paper['name'], paper['surface']))
    print('{}, {}'.format(enlarger['name'], enlarger['lamp']))
    print('f/{} t={}s l={}cm'.format(enlarger['f'], enlarger['t'], enlarger['l']))
    tbl = PrettyTable()
    tbl.field_names = ["Grade", "Y", "M"]
    tbl.align["Grade"] = "l"
    tbl.align["Y"] = "r"
    tbl.align["M"] = "r"
    z = zip(('00', '00.5', '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'),(1.85, 1.7, 1.55, 1.4, 1.28, 1.15, 1.05, 0.95, 0.88, 0.8, 0.73, 0.65, 0.58))
    for i, x in z:
        tbl.add_row([i, int(np.round(filter_yellow(x))), int(np.round(filter_magenta(x)))])
    data = tbl.get_string()
    file_p = file_p.with_suffix('.txt')
    with open(str(file_p), 'w') as f:
        f.write(data)


def densities_extrema(paper):
    # Dmin: a paper has a base refelction density # and processing adds a certain fog level
    #       which together add up to a minimum density
    # Dmax: maximum density possible for a particular paper/processing combination
    #       modern papers can easily reach Dmax of 2.4 or more after toning, so that shadows
    #       become too dark for human detection, thus a practical approach fixes IDMax to 1.89
    # ISO standard defines IDmin as 0.04 above base+fog
    # ISO standard defines IDmax as 90% of Dmax (Dmax == maximum density possible for a particular paper/processing combination) 
    return round(float(paper['b+f']) + 0.04, 2), 1.89


def normalize_wedge(wedge):
    return [round(3.0 - x, 2) for x in ast.literal_eval(wedge['data'])]


def process_data(IDmin, IDmax, relative_log_exposure, data):
    # we must invert relative_log_exposure because InterpolatedUnivariateSpline requires increasing x-values
    relative_log_exposure = relative_log_exposure[::-1]
    x = np.asarray(relative_log_exposure)
    textual_log_exposure_ranges = []
    filter_densities_yellow = []
    filter_densities_magenta = []
    for itm in data:
        reflection_density = ast.literal_eval(itm['data'])
        # we must invert reflection_density because of inversion of relative_log_exposure
        reflection_density = reflection_density[::-1]
        y = np.asarray(reflection_density)
        # interpolate function reflection_density for (x,y)
        reflection_density = InterpolatedUnivariateSpline(x, y)
        # find rel. log exposure for IDmin, use last found root from the list
        pol_dmin = InterpolatedUnivariateSpline(x, (lambda x: reflection_density(x) - IDmin)(x))
        ht = 0 if 0 == pol_dmin.roots().size else round(pol_dmin.roots()[-1], 2)
        # find rel. log exposure for IDmax, use first found root from list
        pol_dmax = InterpolatedUnivariateSpline(x, (lambda x: reflection_density(x) - IDmax)(x))
        hs = round(pol_dmax.roots()[0], 2)
        # compute textual log exposure range
        textual_log_exposure_range = round(hs - ht, 2)
        filter_density_yellow = int(itm['y'])
        filter_density_magenta = int(itm['m'])
        textual_log_exposure_ranges.append(textual_log_exposure_range)
        filter_densities_yellow.append(filter_density_yellow)
        filter_densities_magenta.append(filter_density_magenta)
    # we must invert textual_log_exposure_ranges and filter_desnities because InterpolatedUnivariateSpline requires increasing x-values
    textual_log_exposure_ranges = textual_log_exposure_ranges[::-1]
    filter_densities_yellow = filter_densities_yellow[::-1]
    filter_densities_magenta = filter_densities_magenta[::-1]
    filter_yellow = InterpolatedUnivariateSpline(textual_log_exposure_ranges, filter_densities_yellow)
    filter_magenta = InterpolatedUnivariateSpline(textual_log_exposure_ranges, filter_densities_magenta)
    return textual_log_exposure_ranges, filter_yellow, filter_magenta


def parse_data(file_p):
    config = configparser.ConfigParser()
    config.read(str(file_p))
    paper = config['PAPER']
    enlarger = config['ENLARGER']
    developer = config['DEVELOPER']
    wedge = config['WEDGE']
    data = [(config['M%d' % x]) for x in range(1, 12)]
    return paper, enlarger, developer, wedge, data


def main(file_p):
    paper, enlarger, developer, wedge, data = parse_data(file_p)
    IDmin, IDmax = densities_extrema(paper)
    relative_log_exposure = normalize_wedge(wedge)
    textual_log_exposure_ranges, filter_yellow, filter_magenta = process_data(IDmin, IDmax, relative_log_exposure, data)
    plot_filtration_chart(file_p, paper, enlarger, developer, IDmin, IDmax, textual_log_exposure_ranges, filter_yellow, filter_magenta)
    plot_filtration_tbl(file_p, paper, enlarger, filter_yellow, filter_magenta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    file_p = Path(args.file).resolve()
    assert file_p.exists()
    main(file_p)
