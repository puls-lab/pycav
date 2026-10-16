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
"""Gaussian beam (`Beam`) and resonator (`Cavity`) propagation."""

import numpy as np
import scipy.constants as spc
from pycav.sections import Lens, Mirror, Space, Interface, KerrLens, CustomSection


class Beam(object):
    """Class used for a `Beam` object.

    Attributes
    ----------
    wl : `float`
        wavelength of the beam in meters
    step : `float`
        step width for the propagation in meters, defaults to 1e-3
    _sections : `list`
        list of `Section` objects the beam will go through

    Note: The q-parameter used in this class is defined as follows:
        * A single q-parameter ``q1`` is an `ndarray` with ``q1[:2]`` being the
        saggital and tangential qs, respectively, and ``q1[2]`` is the
        refractive index in which it is defined.
        * a multi q-parameter ``q`` is an `ndarray` of ``q1``-type qs."""

    def __repr__(self):
        s = self._sections[0].__repr__()
        for sec in self._sections[1:]:
            s += '\n' + sec.__repr__()
        return s

    def __init__(self, wl, step=1e-3):
        """init function of `Beam`

        Args
        ----
        wl : `float`
            wavelength of the beam in meters
        step : `float`
            step width for the propagation in meters, defaults to 1e-3"""
        self.wl = wl
        self.step = step
        self._sections = []

    def __getitem__(self, it):
        if isinstance(it, str):
            return self._get_sec_from_idn(it)
        elif isinstance(it, int):  # must be a
            return self._sections[it]
        else:
            raise TypeError('Please enter `str` or `int`!')

    def set_w0(self, w):
        """Set the initial beam waist.

        Args
        ----
        w : `list` of `float` with ``len(w)`` == 2
            the widths of the beam in meters in saggital and tangential
            directions

        Note: This function must be called for a `Beam` before any calculation
        can be performed."""
        self._w0 = np.array(w)
        if hasattr(self, '_q0'):
            del self._q0

    def set_q0(self, q0):
        """Set the initial ``q``.

        Args
        ----
        q : `ndarray` of `complex` with ``len(w)`` == 3
            the q-parameters of the beam in saggital and tangential directions,
            and the refractive index of the material.

        Note: This function must be called for a `Beam` before any calculation
        can be performed, or `set_w0()`."""
        self._q0 = np.array([q0[0], q0[1], 1])
        if hasattr(self, '_w0'):
            del self._w0

    def set_w0R0(self, w0, R0, n=1):
        """Set the initial `q` from the waist and the wavefront radius.

        Args
        ----
        w0 : `double`
             the waist
        R0 : `double`
             the wavefront curvature
        n : `double`
            the refractive index, defaults to 1.
        """
        q0 = 1 / (1 / R0 - 1j * self.wl / n / np.pi / w0**2)
        self.set_q0(np.array([q0, q0, n]))

    def _get_q_init(self):
        """Calculates the initial ``q`` of the beam.

        Returns
        -------
        q1 : `ndarray`
            single q-parameter, see class description for details"""
        if not (hasattr(self, '_w0') or hasattr(self, '_q0')):
            err = """Please set the initial waist radius using ``set_w0()``, or
            set the initial `q` with ``set_q0().``"""
            raise AttributeError(err)
        if hasattr(self, '_w0'):
            q1 = 1j * np.pi * self._w0**2 / self.wl
            return np.array([q1[0], q1[1], 1])
        if hasattr(self, '_q0'):
            return self._q0

    def add_lens(self, f, idn=None):
        """Adds a `Lens` object to ``_sections``.

        Args
        ----
        f : `list` of `float` with ``len(f) == 2``
            the focal length in meters in saggital and tangential
            directions
        idn : `str`
            name of the section, defaults to `None`"""
        self._sections.append(Lens(f=f, idn=idn))

    def add_mirror(self, r=0, aoi=0, idn=None):
        """Adds a `Mirror` object to ``_sections``.

        Args
        ----
        r : `float`
            the radius of curvature of the spherical mirror in meters
        aoi : `float`
            the angle of incidence on the mirror in radians
        idn : `str`
            name of the section, defaults to `None`"""
        self._sections.append(Mirror(r=r, aoi=aoi, idn=idn))

    def add_space(self, d, step=None, idn=None):
        """Adds a `Space` object to ``_sections``.

        Args
        ----
        d : `float`
            the length of the space, ie. the distance, in meters
        step : `float`, defaults to ``Beam.step``
            step width for propagation
        idn : `str`
            name of the section, defaults to `None`"""
        if step is None:
            step = self.step
        self._sections.append(Space(d=d, n=1, step=step, idn=idn))

    def add_custom_section(self, Ms, Mt=None, n=None, idn=None):
        """Adds a `CustomSection` object with user-defined ABCD matrices to
        ``_sections``.

        The section has no length. For standing-wave cavities, the matrix for
        the reverse direction is calculated from the given matrices.

        Args
        ----
        Ms : array_like of shape (2, 2)
            saggital ABCD matrix ``[[A, B], [C, D]]``
        Mt : array_like of shape (2, 2)
            tangential ABCD matrix, defaults to ``Ms``
        n : `float`
            refractive index after the section, defaults to `None`, which
            keeps the refractive index in front of the section. If the
            refractive index changes, the matrices need to include this, ie.
            ``AD - BC = n_in / n_out``.
        idn : `str`
            name of the section, defaults to `None`"""
        if Mt is None:
            Mt = Ms
        self._sections.append(CustomSection(Ms=Ms, Mt=Mt, n=n, idn=idn))

    def add_bulk(self, d, n, aoi=0, step=None, idn=None):
        if idn is None:
            idn = ''
        if step is None:
            step = self.step
        self._sections.append(Interface(ns=[1, n], aoi=aoi, idn=idn + '_Istart'))
        self._sections.append(Space(d=d, n=n, step=step, idn=idn))
        aoi1 = self._sections[-2]._aoi[1]
        self._sections.append(Interface(ns=[n, 1], aoi=aoi1, idn=idn + '_Iend'))

    def add_kerr_medium(self, n, n2, d, P, kerrit=25, step=None, idn=None):
        if self.__class__ == Cavity:
            self.set_kerr_iterations(kerrit)
        self._sections.append(KerrLens(n, n2, d, P, step, idn))

    def add_td(self, d, n, r=1, f=np.inf, aoi=0, step=None, idn=None):
        if idn is None:
            idn = ''
        if step is None:
            step = self.step
        self._sections.append(Interface(ns=[1, n], aoi=aoi, idn=idn + '_Istart'))
        self._sections.append(Space(d=d, n=n, step=step, idn=idn + '_Sstart'))
        aoi1 = self._sections[-2]._aoi[1]
        self._sections.append(Mirror(r=r, aoi=aoi1, idn=idn + '_M'))
        self._sections[-1].n = n
        self._sections.append(Lens(f=(f, f), idn=idn + '_F'))
        self._sections[-1].n = n
        self._sections.append(Space(d=d, n=n, step=step, idn=idn + '_Send'))
        self._sections.append(Interface(ns=[n, 1], aoi=aoi1, idn=idn + '_Iend'))

    def _get_sec_q0(self, q1):
        """Calculates the initial ``q`` parameters for each section.

        Args
        ----
        q : `ndarray`
            single q-parameter, see class description for details"""
        q0s = [q1]
        for sec in self._sections[:-1]:
            if hasattr(sec, '_need_w'):
                sec._set_w(self._get_w(np.array([q1]))[0])
            q1 = sec._apply_M(q1)
            q0s.append(q1)
        return np.array(q0s)

    def _get_w(self, q):
        """Calculates beam width from a series of ``q`` parameters.

        Args
        ----
        q : `ndarray`
            multi q-parameter, see class description for details

        Returns
        -------
        w : `ndarray` of `ndarray`, with ``len(w) == 2``
            widths of the beam that corresponds to the provided ``q``"""
        return np.sqrt(-self.wl / q[:, 2:].real / np.pi / (1 / q[:, :2]).imag)

    def _get_R(self, q):
        """Calculates beam width from a series of q parameters.
        Args
        ----
        q : `ndarray`
            multi q-parameter, see class description for details

        Returns
        -------
        R : `ndarray` of `ndarray`, with ``len(w) == 2``
            widths of the beam that corresponds to the provided ``q``"""
        return 1 / np.real(1 / q[:, :2])

    def get_d(self):
        """Calculates the ``z``-positions of in the beam for the propagation.

        Returns
        -------
        z : `ndarray`
            ``z``-positions of the beam for its propagation"""
        ds = []
        offset = 0
        for sec in self._sections:
            if hasattr(sec, 'd'):
                d = np.arange(0, sec.d, self.step)
                ds.append(offset + d)
                offset += sec.d
            else:
                ds.append(np.array([offset]))
        return np.hstack(ds)

    def get_kerr_num(self):
        for i, sec in enumerate(self._sections):
            if hasattr(sec, '_need_w'):
                return i

    def get_pos(self):
        """Calculates the ``z``-positions of the sections of the beam.

        Returns
        -------
        ds : `ndarray`
            ``z``-positions of sections of the beam"""
        ds = []
        offset = 0
        for sec in self._sections:
            if hasattr(sec, 'd'):
                offset += sec.d
            else:
                ds.append(offset)
        return ds

    def get_pos_plt(self, scale=1):
        """Calculates the ``z``-positions of the sections of the beam and
        returns them in a way to allow easy plotting.

        Args
        ----
        scale : `float`
            value for the scaling of the beam diameter

        Returns
        -------
        ds : look at code for type/format
            ``z``-positions of sections of the beam"""
        ds = self.get_pos()
        return np.array([[ds, ds], [[0] * len(ds), [scale] * len(ds)]])

    def get_qs(self):
        q1 = self._get_q_init()
        if q1 is None:
            # this case is only possible for a cavity, which is not stable
            raise ValueError('Cavity not stable!')
        # get the qs for the beginnings of all sections
        q0s = self._get_sec_q0(q1)
        qs = []
        # propagate the qs and add to list
        for i in range(len(self._sections)):
            qs.append(self._sections[i]._propagate(q0s[i]))
        return qs

    def get_beam(self):
        """Calculates the evolution of the beam.

        Returns
        -------
        d : `ndarray`
            ``z``-positions of the beam
        w : `ndarray` of `ndarray`, with ``len(w) == 2``
            beam width for saggital and tangential directions"""
        # Comment concerning the second output
        # after horizontal stacking, the shape is an array of two arrays, one
        # each for saggital and tangential
        # after transposing the results of _get_w(), the shape is an array of
        # (typically) many arrays of floats of length 2, one for each saggital
        # and tangential
        return self.get_d(), self._get_w(np.vstack(self.get_qs()))

    def get_sec_w(self, idns):
        """Calculate the beam width at certain sections.

        Args
        ----
        idns : `list` of `str`
            List of labels of the sections

        Returns
        -------
        ws : `list` of `ndarray`
            saggital and tangential widths for all sections of interest"""
        nums = [self._get_num_from_idn(idn)[0] for idn in idns]
        qs = self._get_sec_q0(self._get_q_init())[nums]
        return self._get_w(qs)

    def _get_sec_from_idn(self, idn):
        """Returns the section from its name.

        Args
        ----
        idn : `str`
            name of the section

        Returns
        -------
        sec : `Section`
            section that is named ``idn``"""
        secs = []
        for sec in self._sections:
            if sec.idn == idn:
                secs.append(sec)
        return secs

    def _get_num_from_idn(self, idn):
        """Returns the section index in `_sections` from its name.

        Args
        ----
        idn : `str`
            name of the section

        Returns
        -------
        num : `Section` index in ``_sections``
            section that is named ``idn``"""
        nums = []
        for i in range(len(self._sections)):
            if self._sections[i].idn == idn:
                nums.append(i)
        return nums

    def _get_num_w_len(self, inverse=False):
        """Returns the section indexes in `_sections` of all section items that
        have a length.

        Args
        ----
        inverse : `bool`
            Boolean determining whether or not to inverse the search.

        Returns
        -------
        ret : `list` of `int`
            list of indexes"""
        ret = []
        for i in range(len(self._sections)):
            if inverse ^ hasattr(self._sections[i], 'd'):
                ret.append(i)
        return ret

    def _get_idn_w_len(self, inverse=False):
        """Returns the section names in `_sections` of all section items that
        have a length.

        Args
        ----
        inverse : `bool`
            Boolean determining whether or not to inverse the search.

        Returns
        -------
        ret : `list` of `str`
            list of names"""
        ret = []
        for i in range(len(self._sections)):
            if inverse ^ hasattr(self._sections[i], 'd'):
                ret.append(self._sections[i].idn)
        return ret

    def has_kerr(self):
        for sec in self._sections:
            if hasattr(sec, '_need_w'):
                return True

    def vary(self, vsec, osec):
        """Vary a single parameter of a section and return the beam width on
        another section.

        Args
        ----
        vsec : (`vary_section.idn`, `vary_section.par`, `ndarray`)
            This argument is an iterable with the `idn`` (`str`) of the
            section that should be varied, the variation parameter ``par`` of
            that section, and the range in which it should be varied.
        osec : `str`
            This argument is the ``idn`` of the section, where the beam width
            is requested.

        Returns
        -------
        w : `ndarray`
            saggital and tangential beam widths on the desired section

        Example
        -------
        variation of the radius of curvature of M1, observation of the spot
        size on M1
        .. code:: python

            import pycav
            import numpy as np
            b = pycav.Beam(1e-6)
            b.set_w0([1e-3, 1e-3])
            b.add_space(1)
            b.add_mirror(1, 0, 'M1')
            b.add_space(1)
            b.add_mirror(1, 0, 'M2')
            wv = b.vary(('M1', 'r', np.linspace(1, 2, 1000)), 'M2')"""
        vidn, pstr, rng = vsec
        vsecs = self._get_sec_from_idn(vidn)
        p0 = getattr(vsecs[0], pstr)
        osec = self._get_num_from_idn(osec)[0]
        qs = np.empty((len(rng), 3), dtype=complex)
        for i, p in enumerate(rng):
            # set the parameter
            for vsec in vsecs:
                setattr(vsec, pstr, p)
            # get q, ie. check if cavity stable
            qi = self._get_q_init()
            if qi is None:
                # cavity is not stable => use NAN, so that the returned arrays
                # are complete, ie. don't skip values, and don't create
                # shortcuts
                qs[i] = np.array([np.nan, np.nan, np.nan])
            else:
                # only use qs for desired sections
                q0s = self._get_sec_q0(qi)
                qs[i] = q0s[osec]
        # reset to original value
        for vsec in vsecs:
            setattr(vsec, pstr, p0)
        return self._get_w(qs)

    def vary_2D(self, vsecs, osec):
        """Vary two parameters of two sections, sections can be the same, and
        return the beam width on another section.

        Args
        ----
        vsec : [(`vary_section1.idn1`, `vary_section1.par1`, `ndarray`),
                (`vary_section2.idn2`, `vary_section2.par2`, `ndarray`)
            Similar syntax to `self.vary()`, but a list of two `vsec` items is
            used.
        osec : `str`
            This argument is the ``idn`` of the section, where the beam width
            is requested.

        Returns
        -------
        w : `ndarray`
            saggital and tangential beam widths on the desired section
            With ``ws, wt = w``, one can directly make a contour plot using
            ``ws`` or ``wt``.

        Example
        -------
        variation of the radius of curvature of M1, observation of the spot
        size on M1
        .. code:: python

            import pycav
            import numpy as np
            b = pycav.Beam(1e-6)
            b.set_w0([1e-3, 1e-3])
            b.add_space(1, 'd1')
            b.add_mirror(1, 0, 'M1')
            b.add_space(1)
            b.add_mirror(1, 0, 'M2')
            ws, wt = b.vary_2D([('d1', 'd', np.linspace(0, 1, 100)),
                                ('M1', 'r', np.linspace(1, 2, 100))], 'M2')"""
        (vidn0, pstr0, rng0), (vidn1, pstr1, rng1) = vsecs
        vsecs0 = self._get_sec_from_idn(vidn0)
        vsecs1 = self._get_sec_from_idn(vidn1)
        dat = []
        p0 = getattr(vsecs1[0], pstr1)
        for v in rng1:
            # setting attribute of second vsection
            for vsec1 in vsecs1:
                setattr(vsec1, pstr1, v)
            # varying the first vsection and appending data
            dat.append(self.vary(vsecs[0], osec))
        # setting data of second vsection to original value
        for vsec1 in vsecs1:
            setattr(vsec1, pstr1, p0)
        dat = np.array(dat)
        # return data in a convenient shape for direct contour plotting
        return [dat[:, :, 0], dat[:, :, 1]]

    def vary_multi(self, vsec, osecs):
        """Vary a single parameter of a section and return the beam widths on
        several other sections.

        Args
        ----
        vsec : (`vary_section.idn`, `vary_section.par`, `ndarray`)
            This argument is an iterable with the `idn`` (`str`) of the
            section that should be varied, the variation parameter ``par`` of
            that section, and the range in which it should be varied.
        osecs : `list` of `str`
            This argument is a `list` of the ``idn`` of sections, for which the
            beam width is requested.

        Returns
        -------
        w : `ndarray`
            saggital and tangential beam widths on the desired sections in the
            following order: w[0] =
                                    [w_osec[0]_sag, w_osec[0]_tan,
                                     w_osec[1]_sag, w_osec[1]_tan, ...]
            This allows for easy plotting using ``plt.plot(rng, w)``.

        Example
        -------
        variation of the radius of curvature of M1, observation of the spot
        size on M1
        .. code:: python

            import pycav
            import numpy as np
            b = pycav.Beam(1e-6)
            b.set_w0([1e-3, 1e-3])
            b.add_space(1)
            b.add_mirror(1, 0, 'M1')
            b.add_space(1)
            b.add_mirror(1, 0, 'M2')
            b.add_space(1)
            b.add_mirror(1, 0, 'M3')
            wv = b.vary(('M1', 'r', np.linspace(1, 2, 1000)), ['M2', 'M3'])"""
        vidn, pstr, rng = vsec
        vsecs = self._get_sec_from_idn(vidn)
        p0 = getattr(vsecs[0], pstr)
        # determine section indexes
        osecsn = [self._get_num_from_idn(sec)[0] for sec in osecs]
        qs = []
        for p in rng:
            # set attribute of variation section
            for vsec in vsecs:
                setattr(vsec, pstr, p)
            qi = self._get_q_init()
            if qi is None:
                qs.append(np.array([np.nan, np.nan, np.nan] * len(osecs)))
            else:
                q0s = self._get_sec_q0(qi)
                qs.append(q0s[osecsn].flatten())
        # reset to original value
        for vsec in vsecs:
            setattr(vsec, pstr, p0)
        qs = np.array(qs).reshape((-1, 3))
        return self._get_w(qs).reshape((len(rng), 2 * len(osecs)))


class Cavity(Beam):
    """Class used for a `Cavity` object, which is strongly based on `Beam`.

    Attributes (check superclass for more)
    --------------------------------------
    _ring : `bool`
        determines if the cavity is a ring cavity"""

    def __init__(self, wl, step=1e-3, ring=False):
        """init function of `Cavity`

        Args
        ----
        wl : `float`
            wavelength of the beam in meters
        step : `float`
            step width for the propagation in meters, defaults to 1e-3
        ring : `bool`, defaults to ``False``
            determines if the cavity is a ring cavity"""
        super().__init__(wl, step)
        self._ring = ring

    def _get_M(self):
        """Calculates the matrix of the cavity."""
        M = np.matrix(np.diag(np.ones(4)))
        # go through all cavity sections
        for sec in self._sections:
            M = sec.M.dot(M)
        # if standing wave cavity, go back to beginning
        if not self._ring:
            for sec in self._sections[-2:0:-1]:
                # use the reverse direction matrix Mrev
                M = sec.Mrev.dot(M)
        return M

    def _get_q_init(self):
        """Calculates the initial ``q`` of the cavity, if possible.

        Returns
        -------
        q : `ndarray` of two `complex`
            qs for saggital and tangential directions"""
        if not self.has_kerr():
            return self._get_q_init_single()
        else:
            return self._get_q_init_kerr()

    def _get_q_init_kerr(self):
        knum = self.get_kerr_num()
        Pf = self._sections[knum].P
        w = None
        for P in np.sqrt(np.linspace(0, Pf**2, self._kerrit)):
            self._sections[knum].P = P
            if P != 0:
                self._get_sec_q0(q0)  # this also set w0 for the Kerr medium
            q0 = self._get_q_init_single()
            if q0 is None:
                self._sections[knum].P = Pf
                return None
        return q0

    def _get_q_init_single(self):
        M = self._get_M()
        try:
            q0s = np.roots((M[1, 0], M[1, 1] - M[0, 0], -M[0, 1]))[0]
            q0t = np.roots((M[3, 2], M[3, 3] - M[2, 2], -M[2, 3]))[0]
        except IndexError:
            # this can happen if only the constant term in the polynomial is
            # non-zero, and there is no solution
            q0s = q0t = np.array([0])
        q0 = np.hstack((q0s, q0t))
        if np.any(q0.imag == 0):
            return None
        else:
            q0 = q0.real + 1j * abs(q0.imag)
            return np.array([q0[0], q0[1], 1])

    def set_kerr_iterations(self, kerrit):
        self._kerrit = kerrit

    def get_reprate(self):
        """Calculates the repetition rate of the cavity.

        Returns
        -------
        f : `float`
            repetion rate of the cavity"""
        if self._ring:
            return spc.c / self.get_d()[-1]
        else:
            return spc.c / self.get_d()[-1] / 2

    def get_gouy_phase(self):
        M = self._get_M()
        gouy_s = 2 * np.arccos(np.sign(M[0, 1]) * np.sqrt((M[0, 0] + M[1, 1] + 2) / 4))
        gouy_t = 2 * np.arccos(np.sign(M[2, 3]) * np.sqrt((M[2, 2] + M[3, 3] + 2) / 4))
        return gouy_s, gouy_t
