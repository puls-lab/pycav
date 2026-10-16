# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2017-2026 Dr. Martin Saraceno
#
# This file is part of pycav.
#
# pycav is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pycav is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pycav. If not, see <https://www.gnu.org/licenses/>.
#
# Additional terms under section 7(b) of the GNU GPL apply; see the NOTICE file.
"""Interactive matplotlib cavity designer (`CavityDesigner`)."""

import matplotlib.pyplot as plt
import numpy as np

label_units = {
    'd': ' / m',
    'f': ' / m',
    'f1': ' / m',
    'fs': ' / m',
    'ft': ' / m',
    'r': ' / m',
    'aoi': ' / rad',
    'P': ' / W',
}

la = {'ha': 'left', 'va': 'bottom', 'rotation': 'vertical', 'fontsize': 8}


class CavityDesigner(object):
    """Class to simplify the building of laser cavities. It enables to
    conveniently click into the graphs to vary the cavity parameters.

    Attributes
    ----------
    cav : `Cavity`
        cavity that is simulated
    varpars : `list` of (section.idn, section.par, range)
        sections, parameters, and range that should be varied; one additional
        graph per element
    elsofin : `list` of `str`
        section names of which the beam size should be calculated in the plots
        corresponding to `varpars`
    f : `plt.figure`
        figure object
    axs : `plt.Axes`
        axes objects
    cplts : `plt.plot`
        plot objects of the cavity, including bars for sections
    plts : `plt.plot`
        plot objects of the variation plots
    inds : `plt.plot`
        plot object for the current position indicators
    seclabs : `plt.text`
        text objects for the sections in the beam propagation plot"""

    def __init__(self, cav, varpars, elsofin, figsize=(9, 6)):
        self.cav = cav
        self.varpars = varpars
        # els are the sections where text will be displayed in the beam graph
        self.els = dict(
            zip(
                self.cav._get_idn_w_len(inverse=True),
                self.cav._get_num_w_len(inverse=True),
            )
        )
        self.elsofin = elsofin
        self.init_graphs()

    def init_graphs(self, figsize=(9, 6)):
        # set up figure and axes
        self.f = plt.figure(figsize=figsize)
        self.axs = []
        self.cplts = []
        self.plts = []
        self.inds = []
        self.seclabs = []
        # set up the number of graphs and axes
        numrows = int(np.ceil(len(self.varpars) / 2))
        self.cax = plt.subplot2grid((numrows + 1, 2), (0, 0), colspan=2)
        # title will contain the repetion rate
        self.cax.set_title('', fontsize=10)
        for i in range(len(self.varpars)):
            self.axs.append(plt.subplot2grid((numrows + 1, 2), (int(i / 2) + 1, i % 2)))
        self.set_axes()
        # make dummy plots for the cavity (beam evolution of sag and tang)
        self.cplts.append(self.cax.plot([np.nan, np.nan], np.zeros((2, 2))))
        # make dummy plot for the vertical bars for the section elements
        num = len(self.cav.get_pos_plt()[0, 0])
        self.cplts.append(self.cax.plot([np.nan, np.nan], np.zeros((2, num)), 'k'))
        # make labels for beam sizes on elements w/o length
        for i in self.cav._get_num_w_len(inverse=True):
            self.seclabs.append(self.cax.text(0, 0, self.cav._sections[i].idn, **la))
        self.set_lines(self.cplts[0])
        # update the cavity
        self.update_cav()
        # make dummy plots for the variation parameters
        for ax in self.axs:
            self.plts.append(
                ax.plot([np.nan, np.nan], np.zeros((2, 2 * len(self.elsofin))))
            )
            self.inds.append(ax.plot([np.nan, np.nan], [np.nan, np.nan], 'k:'))
        for pltobj in self.plts:
            self.set_lines(pltobj)
        # update the variation plots
        for i in range(len(self.varpars)):
            self.update_num(i)
        self.f.tight_layout()
        # set up clicky-clicky
        self.f.canvas.mpl_connect('button_press_event', self.onclick)

    def update_cav(self):
        """This function updates the cavity plot, ie. the beam evolution."""
        # update the beam evolution plots (sag and tang)
        d, w = self.cav.get_beam()
        self.cplts[0][0].set_data((d, w.T[0]))
        self.cplts[0][1].set_data((d, w.T[1]))
        if hasattr(self.cav, 'get_reprate'):
            s = 'rep. rate: {:0.2e} Hz'.format(self.cav.get_reprate())
        else:
            s = ''
        self.cax.title.set_text(s)
        # update the bars for the elements and the text with beam sizes
        for i, po in enumerate(self.cav.get_pos_plt(w.max()).T):
            self.cplts[1][i].set_data(po.T)
            self.seclabs[i].set_position((po[0, 0], 0))
        d = self.cav.get_sec_w(self.els.keys())
        for i, k in enumerate(self.els.keys()):
            s = '{:}:\nws={:0.2e} m\nwt={:0.2e} m'.format(k, *d[i])
            self.seclabs[i].set_text(s)
        # reset the limits of the plot
        self.cax.relim()
        self.cax.autoscale_view(True, True, True)

    def update_num(self, num, xdata=None):
        """Update a selected variation plot.

        Args
        ----
        num : `int`
            index of the variation parameter as given in ``varpars``
        xdata : `float`, defaults to ``None``
            new value for the parameter"""
        idn, par, rng = self.varpars[num]
        celnum = self.cav._get_num_from_idn(idn)[0]
        if xdata is None:
            # if None, use original value from the cavity
            val = getattr(self.cav._sections[celnum], par)
            self.inds[num][0].set_xdata([val, val])
        # get the beam sizes for the variation
        dat = self.cav.vary_multi(self.varpars[num], self.elsofin)
        # update the values in the plots of the appropriate axis
        for i, p in enumerate(self.plts[num]):
            p.set_data(rng, dat.T[i])
        # adjust the length of the vertical bar
        self.inds[num][0].set_ydata([0, dat[np.isfinite(dat)].max()])
        # relimit the graph
        self.axs[num].relim()
        self.axs[num].autoscale_view(True, True, True)

    def set_lines(self, pltobj):
        """Set the lines of ``plotobj`` to look nice.

        Args
        ----
        pltobj : `plt.plot`
            plot object to alter the lines of"""
        for i in range(int(len(pltobj) / 2)):
            pltobj[2 * i].set_color('C' + str(i))
            pltobj[2 * i + 1].set_color('C' + str(i))
            pltobj[2 * i + 1].set_linestyle('--')

    def set_axes(self):
        """Set up the axes of the Designer."""
        self.cax.set_xlabel('cavity length / m')
        self.cax.set_ylabel('beam width / m')
        for i in range(len(self.varpars)):
            ut = label_units[self.varpars[i][1]]
            ylabelstr = (
                'variation: ' + self.varpars[i][0] + '(' + self.varpars[i][1] + ')' + ut
            )
            self.axs[i].set_xlabel(ylabelstr)
            self.axs[i].set_ylabel('beam width / m')

    def get_plt_num(self, inax):
        """Returns the index of the input axis.

        Args
        ----
        inax : `plt.Axis`
            axis of which the index is desired"""
        for i, ax in enumerate(self.axs):
            if ax == inax:
                return i

    def onclick(self, event):
        """Event handler for clicking events."""
        # get axis and index of axis
        ax = event.inaxes
        pltnum = self.get_plt_num(event.inaxes)
        # move bar indicator on clicked axis to clicked position
        self.inds[pltnum][0].set_xdata((event.xdata, event.xdata))
        # get indexes and paramter and set to new (clicked) value
        idn, par, _ = self.varpars[pltnum]
        celnums = self.cav._get_num_from_idn(idn)
        for celnum in celnums:
            setattr(self.cav._sections[celnum], par, event.xdata)
        # update cavity plot
        self.update_cav()
        # update other variation plots
        for i in range(len(self.varpars)):
            if i == pltnum:
                continue
            self.update_num(i, event.xdata)
