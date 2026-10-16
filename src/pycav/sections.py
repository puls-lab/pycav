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
"""Optical sections (lenses, mirrors, spaces, interfaces) for beam propagation."""

import numpy as np


class Section(object):
    """Base class `Section` for sections for cavity simulations.

    Attributes
    ----------
    idn : `str`
        identifier of the section
    M : `np.matrix`
        4x4 matrix where the top-left 2x2 matrix represents the saggital, and
        the bottom-right 2x2 matrix the tangential ABCD matrix of the section.
    step : `float`
        step width for propagation"""

    def __repr__(self):
        s = '{:<10} idn = {:}'.format(self.__class__.__name__ + ':', self.idn)
        return s

    def __init__(self, n, step, idn):
        """Init function.

        Args
        ----
        n : `float`
            refractive index
        step : `float`
            step width for propagation
        idn : `str`
            identifier or name of the section"""
        self.M = np.matrix(np.diag(np.ones(4)))
        self.idn = idn
        self.step = step
        self.n = n

    def get_n(self):
        return self._n

    def set_n(self, n):
        self._n = n

    n = property(get_n, set_n)

    def _propagate(self, q1):
        """Propagates an initial input q through the section.

        Args
        ----
        q1 : `ndarray`
            single q-parameter

        Returns
        -------
        q : `ndarray`
            multi q-parameter"""
        # make a propagation if section has a length
        if hasattr(self, 'd'):
            prop = np.arange(0, self.d, self.step)
            prop = np.array([prop, prop, np.zeros(len(prop))]).T
            return q1 + prop
        else:
            return np.array([q1])

    def _apply_M(self, q1):
        """Apply the section matrix to an input q.

        Args
        ----
        q1 : `q` single parameters
            `q` parameter at the beginning of the section

        Returns
        --------
        q : `ndarray` two `complex`
            q parameters at the end of the section"""
        q1s, q1t, _ = q1
        q1s = (self.M[0, 0] * q1s + self.M[0, 1]) / (self.M[1, 0] * q1s + self.M[1, 1])
        q1t = (self.M[2, 2] * q1t + self.M[2, 3]) / (self.M[3, 2] * q1t + self.M[3, 3])
        return np.array([q1s, q1t, self.n])


class Lens(Section):
    """Class for a `Lens` section.

    Properties
    ----------
    f : [`float`, `float`]
        get/set ``fs`` and ``ft``
    f1 : `float`
        get/set ``f``, which needs to be identical for ``fs`` and ``ft`` for
        the getter
    fs : `float`
        get/set ``fs``
    ft : `float`
        get/set ``ft``

    Note: For all setters, the section matrix is updated, too."""

    def __repr__(self):
        s = super().__repr__()
        s += ', fs = {:0.4f} m, ft = {:0.4f} m'.format(self.f[0], self.f[1])
        return s

    def __init__(self, f, idn):
        """Init function.

        Args
        ----
        f : [``fs``, ``ft``]
            list of saggital and tangential focal lengths
        idn : `str`
            label of the lens"""
        super().__init__(1, 1, idn)
        self.f = f

    def get_f(self):
        """Property that gets/sets the focal lengths as [``fs``, ``ft``]"""
        return self._f

    def set_f(self, f):
        self._f = f
        self._update_M()

    f = property(get_f, set_f)

    def get_f1(self):
        """Property that gets/sets the focal length ``f``. For the getter,
        ``fs`` and ``ft`` need to be identical."""
        if self._f[0] != self._f[1]:
            raise ValueError('f[0] != f[1]')
        return self._f[0]

    def set_f1(self, f1):
        self._f = (f1, f1)
        self._update_M()

    f1 = property(get_f1, set_f1)

    def get_fs(self):
        """Property that gets/sets the saggital focal length ``fs``."""
        return self._f[0]

    def set_fs(self, fs):
        self._f[0] = fs
        self._update_M()

    fs = property(get_fs, set_fs)

    def get_ft(self):
        """Property that gets/sets the tangential focal length ``fs``."""
        return self._f[1]

    def set_ft(self, ft):
        self._f[1] = ft
        self._update_M()

    ft = property(get_ft, set_ft)

    def _update_M(self):
        """Function that updates the matrix according to class attributes."""
        self.M = np.matrix(
            [
                [1, 0, 0, 0],
                [-1 / self._f[0], 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1 / self._f[1], 1],
            ]
        )
        self.Mrev = self.M


class KerrLens(Lens):
    def __init__(self, n, n2, d, P, step, idn):
        super().__init__([np.inf, np.inf], idn)
        self._need_w = True
        self._w = None
        self._n2 = n2
        self._d = d
        self._P = P

    def _get_w(self):
        return self._w

    def _set_w(self, w):
        self._w = w
        self._update_f(w=w)

    def get_P(self):
        return self._P

    def set_P(self, P):
        self._P = P
        self._update_f(P=P)

    P = property(get_P, set_P)

    def _update_f(self, **kwargs):
        w = kwargs['w'] if 'w' in kwargs else self._w
        P = kwargs['P'] if 'P' in kwargs else self._P
        # Next line correct?!? factor 4 or 2?
        if w is not None:
            self._f = 1 / (8 * self._n2 * self._d * P / np.pi / w**4)
        else:
            self._f = np.array([np.inf, np.inf])
        self._update_M()


class Mirror(Section):
    """Class for a `Mirror` section.

    Properties
    ----------
    r : float`
        get/set the radius of curvature
    aoi : `float`
        get/set angle of incidence, given in radians

    Note: For all setters, the section matrix is updated, too."""

    def __repr__(self):
        s = super().__repr__()
        s += ', r = {:0.4f} m, aoi = {:0.4f} deg'.format(self.r, np.rad2deg(self.aoi))
        return s

    def __init__(self, r, aoi, idn):
        """Init function.

        Args
        ----
        r : `float`
            radius of curvature
        aoi : `float`
            angle of incidence in radians
        idn : `str`
            label of the mirror"""
        super().__init__(1, 1, idn)
        self._r = r
        self._aoi = aoi
        self.r = r
        self.aoi = aoi

    def get_r(self):
        """Property that gets/sets the radius of curvature."""
        return self._r

    def set_r(self, r):
        self._r = r
        if r == 0:
            self.M = np.matrix(np.diag(np.ones(4)))
        else:
            self.M = np.matrix(
                [
                    [1, 0, 0, 0],
                    [-2 / r * np.cos(self._aoi), 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, -2 / r / np.cos(self._aoi), 1],
                ]
            )
        self.Mrev = self.M

    r = property(get_r, set_r)

    def get_aoi(self):
        """Property that gets/sets the angle of incidence."""
        return self._aoi

    def set_aoi(self, aoi):
        self._aoi = aoi
        if self._r == 0:
            self.M = np.matrix(np.diag(np.ones(4)))
        else:
            self.M = np.matrix(
                [
                    [1, 0, 0, 0],
                    [-2 / self._r * np.cos(aoi), 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, -2 / self._r / np.cos(aoi), 1],
                ]
            )
        self.Mrev = self.M

    aoi = property(get_aoi, set_aoi)


class Space(Section):
    """Class for a `Space` section.

    Properties
    ----------
    d : float`
        get/set the length of the section

    Note: For the setter, the section matrix is updated, too."""

    def __repr__(self):
        s = super().__repr__()
        s += ', d = {:0.4f} m, n = {:0.4f}'.format(self.d, self.n)
        return s

    def __init__(self, d, n, step, idn):
        """Init function.

        Args
        ----
        d : `float`
            length of the section
        n : `float`
            refractive index
        step : `float`
            step width for propagation
        idn : `str`, defaults to ``None``
            label of the space section"""
        super().__init__(n, step, idn)
        self.d = d

    def get_d(self):
        """Property that gets/sets the length of the section."""
        return self._d

    def set_d(self, d):
        self._d = d
        self.M = np.matrix([[1, d, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
        self.Mrev = self.M

    d = property(get_d, set_d)


class Interface(Section):
    def __repr__(self):
        s = super().__repr__()
        s += ', n0 = {:0.4f}, n1 = {:0.4f}'.format(self.ns[0], self.ns[1])
        return s

    def __init__(self, ns, aoi, idn):
        # set refractive index to final refractive index
        super().__init__(ns[1], 1, idn)
        aoi1 = np.arcsin(ns[0] * np.sin(aoi) / ns[1])
        self._aoi = np.array([aoi, aoi1])
        self._angleCorr = np.cos(aoi) / np.cos(aoi1)
        self.ns = ns

    def get_n(self):
        return self._ns

    def set_n(self, ns):
        self._ns = ns
        self.M = np.matrix(
            [
                [1, 0, 0, 0],
                [0, ns[0] / ns[1], 0, 0],
                [0, 0, 1 / self._angleCorr, 0],
                [0, 0, 0, ns[0] / ns[1] * self._angleCorr],
            ]
        )
        self.Mrev = np.matrix(
            [
                [1, 0, 0, 0],
                [0, ns[1] / ns[0], 0, 0],
                [0, 0, self._angleCorr, 0],
                [0, 0, 0, ns[1] / ns[0] / self._angleCorr],
            ]
        )

    ns = property(get_n, set_n)


class CustomSection(Section):
    """Class for a `CustomSection` with user-defined ABCD matrices.

    The section has no length; it only applies its matrices to the beam. The
    matrix for the reverse direction, which is needed for standing-wave
    cavities, is calculated as ``[[D, B], [C, A]] / (AD - BC)``.

    Properties
    ----------
    Ms : `ndarray` of shape (2, 2)
        get/set the saggital ABCD matrix
    Mt : `ndarray` of shape (2, 2)
        get/set the tangential ABCD matrix

    Note: For all setters, the section matrix is updated, too."""

    def __init__(self, Ms, Mt, n, idn):
        """Init function.

        Args
        ----
        Ms : array_like of shape (2, 2)
            saggital ABCD matrix
        Mt : array_like of shape (2, 2)
            tangential ABCD matrix
        n : `float` or `None`
            refractive index after the section, `None` keeps the refractive
            index in front of the section
        idn : `str`
            label of the section"""
        super().__init__(n, 1, idn)
        self._Ms = self._check_M(Ms)
        self._Mt = self._check_M(Mt)
        self._update_M()

    def get_Ms(self):
        """Property that gets/sets the saggital ABCD matrix."""
        return self._Ms

    def set_Ms(self, Ms):
        self._Ms = self._check_M(Ms)
        self._update_M()

    Ms = property(get_Ms, set_Ms)

    def get_Mt(self):
        """Property that gets/sets the tangential ABCD matrix."""
        return self._Mt

    def set_Mt(self, Mt):
        self._Mt = self._check_M(Mt)
        self._update_M()

    Mt = property(get_Mt, set_Mt)

    @staticmethod
    def _check_M(M):
        M = np.array(M, dtype=float)
        if M.shape != (2, 2):
            raise ValueError(
                'ABCD matrix must have shape (2, 2), not {}'.format(M.shape)
            )
        if np.linalg.det(M) == 0:
            raise ValueError('ABCD matrix must not be singular')
        return M

    @staticmethod
    def _reverse(M):
        """Returns the ABCD matrix for propagation in the reverse direction."""
        (A, B), (C, D) = M
        return np.array(((D, B), (C, A))) / (A * D - B * C)

    def _update_M(self):
        """Function that updates the matrices according to class attributes."""
        self.M = np.matrix(np.zeros((4, 4)))
        self.M[:2, :2] = self._Ms
        self.M[2:, 2:] = self._Mt
        self.Mrev = np.matrix(np.zeros((4, 4)))
        self.Mrev[:2, :2] = self._reverse(self._Ms)
        self.Mrev[2:, 2:] = self._reverse(self._Mt)

    def _apply_M(self, q1):
        """Like `Section._apply_M`, but keeps the refractive index of ``q1`` if
        ``n`` is `None`."""
        q = super()._apply_M(q1)
        if self.n is None:
            return np.array([q[0], q[1], q1[2]])
        return q
