from __future__ import division, print_function
import math
from builtins import zip, range
import numpy as np
from scipy.integrate import quad
from fast3tree import fast3tree, get_distance

__all__ = ['projected_correlation', 'correlation3d', 'get_pairs']


def _fast_histogram1d(a, bins):
    """
    Note: `a` is modified in place!
    """
    a.sort()
    return np.ediff1d(np.searchsorted(a, bins))


def _yield_periodic_points(center, dcorner1, dcorner2, box_size):
    cc = np.array(center)
    flag = (cc+dcorner1 < 0).astype(int) - (cc+dcorner2 >= box_size).astype(int)
    cp = cc + flag*box_size
    for j in range(1 << len(cc)):
        for i in range(len(cc)):
            if j >> i & 1 == 0:
                cc[i] = center[i]
            elif flag[i]:
                cc[i] = cp[i]
            else:
                break
        else:
            yield cc.copy()


def _corner_area(x, y):
    a = math.sqrt(1.0-x*x)-y
    b = math.sqrt(1.0-y*y)-x
    theta = math.asin(math.sqrt(a*a+b*b)*0.5)*2.0
    return (a*b + theta - math.sin(theta))*0.5

def _segment_without_corner_area(x, r):
    half_chord = math.sqrt(1.0-x*x)
    return math.acos(x) - x*half_chord \
            - quad(_corner_area, 0, min(half_chord, 1.0/r), (x,))[0]*r

def _overlapping_circular_areas(r):
    if r*r >= 2.0:
        return 1.0
    elif r <= 0:
        return 0.0
    return (math.pi - quad(_segment_without_corner_area, 0, min(1, 1.0/r), \
            (r,))[0]*4.0*r)*r*r

_overlapping_circular_areas_vec = np.vectorize(_overlapping_circular_areas, [float])


def _jackknife_2d_random(rbins, box_size, jackknife_nside):
    side_length = box_size/float(jackknife_nside)
    square_area = 1.0/float(jackknife_nside*jackknife_nside)
    rbins_norm = rbins/side_length
    annulus_areas = np.ediff1d(_overlapping_circular_areas_vec(rbins_norm))
    annulus_areas /= np.ediff1d(rbins_norm*rbins_norm)*math.pi
    return 1.0 - square_area * (2.0 - annulus_areas)


def _get_pairs_max_sphere(points1, points2, max_radius, periodic_box_size=None, indices1=None, indices2=None):

    assert max_radius > 0

    is_periodic = False
    box_size = -1
    if periodic_box_size is not None:
        is_periodic = True
        box_size = float(periodic_box_size)
        assert box_size > 0
        if max_radius*2.0 > box_size:
            print('[Warning] box too small!')

    with fast3tree(points2, indices2) as tree:
        del points2, indices2
        if box_size > 0:
            tree.set_boundaries(0, box_size)
        iter_points1 = (enumerate(points1) if indices1 is None else zip(indices1, points1))
        del points1, indices1
        for i, p in iter_points1:
            indices, pos = tree.query_radius(p, max_radius, is_periodic, 'both')
            if len(indices):
                pos = get_distance(p, pos, box_size)
                yield i, indices, pos


def _get_pairs_max_box(points1, points2, max_distances, periodic_box_size=None, indices1=None, indices2=None):

    max_distances = np.asanyarray(max_distances)
    max_distances_neg = max_distances * (-1.0)
    assert np.all(max_distances - max_distances_neg > 0)

    is_periodic = False
    if periodic_box_size is not None:
        is_periodic = True
        box_size = float(periodic_box_size)
        assert box_size > 0
        assert np.all(max_distances - max_distances_neg < box_size)

    with fast3tree(points2, indices2) as tree:
        del points2, indices2
        iter_points1 = (enumerate(points1) if indices1 is None else zip(indices1, points1))
        del points1, indices1
        for i, p in iter_points1:
            p_iter = _yield_periodic_points(p, max_distances_neg, max_distances, box_size) if is_periodic else [p]
            for pp in p_iter:
                indices, pos = tree.query_box(pp + max_distances_neg, pp + max_distances, output='both')
                if len(indices):
                    pos -= pp
                    yield i, indices, pos


def _reduce_2d_distance_square(pos):
    """
    Note: `pos` is modified in place.
    """
    pos = pos[:,:2]
    pos *= pos
    pos[:,0] += pos[:,1]
    return pos[:,0]


def get_pairs(points1, points2, max_radius, max_dz=None, periodic_box_size=None,
              id1_label='id1', id2_label='id2', dist_label='d', can_swap_points=True,
              indices1=None, indices2=None, wrapper_function=None):
    """
    Identify all pairs within a sphere or a cylinder.

    Parameters
    ----------
    points1 : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
    points2 : array_like or None
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        If set to None, do auto-correlation
    max_radius : float
        Find pairs within this radius
    max_dz : float or None
    periodic_box_size : float or None
    """

    if points2 is not None and can_swap_points and len(points2) > len(points1):
        points1, points2 = points2, points1
        id1_label, id2_label = id2_label, id1_label


    is_autocorrelation = False
    if points2 is None:
        points2 = points1
        if indices2 is None and indices1 is not None:
            indices2 = indices1
        is_autocorrelation = True


    if max_dz is None:
        function_to_call = _get_pairs_max_sphere
        max_distances = max_radius
        if wrapper_function is None:
            def wrapper_function(iter_input):
                for i, j_arr, d_arr in iter_input:
                    if is_autocorrelation:
                        flag = (j_arr > i)
                        j_arr = j_arr[flag]
                        d_arr = d_arr[flag]
                        del flag
                    for j, d in zip(j_arr, d_arr):
                        yield i, j, d
    else:
        function_to_call = _get_pairs_max_box
        max_distances = np.array([max_radius, max_radius, max_dz])
        if wrapper_function is None:
            def wrapper_function(iter_input):
                for i, j_arr, d_arr in iter_input:
                    d_arr = _reduce_2d_distance_square(d_arr)
                    np.sqrt(d_arr, out=d_arr)
                    flag = (d_arr < max_radius)
                    if is_autocorrelation:
                        flag &= (j_arr > i)
                    j_arr = j_arr[flag]
                    d_arr = d_arr[flag]
                    del flag
                    for j, d in zip(j_arr, d_arr):
                        yield i, j, d

    iterator = wrapper_function(function_to_call(points1, points2, max_distances,
            periodic_box_size, indices1, indices2))
    del points1, points2, indices1, indices2

    output_dtype = np.dtype([(id1_label, np.int), (id2_label, np.int), (dist_label, np.float)])
    return np.fromiter(iterator, output_dtype)


def _check_points(points):
    points = np.asarray(points)
    s = points.shape
    if len(s) != 2 or s[1] != 3:
        raise ValueError('`points` must be a 2-d array with last dim=3')
    return points, s[0]


def _check_rbins(rbins):
    rbins = np.asarray(rbins)
    assert (np.ediff1d(rbins) > 0).all(), '`rbins` must be an increasing array'
    assert rbins[0] >= 0, '`rbins must be all positive'
    return rbins


def get_random_pair_counts(n_points, box_size, rbins, zmax=None):
    rbins = np.asarray(rbins)
    assert (np.ediff1d(rbins) > 0).all()

    density = n_points * n_points / (box_size**3.0)

    if zmax is None:
        volume = np.ediff1d(rbins**3.0) * (np.pi*4/3)
    else:
        volume = np.ediff1d(rbins*rbins) * np.pi * zmax * 2.0

    return density * volume


def get_jackknife_ids(points, box_size, jackknife_nside):
    jackknife_nside = int(jackknife_nside)
    jack_ids  = np.floor(np.remainder(points[:,0], box_size)\
            / box_size * jackknife_nside).astype(int)
    jack_ids += np.floor(np.remainder(points[:,1], box_size)\
            / box_size * jackknife_nside).astype(int) * jackknife_nside
    return(jack_ids)


def projected_correlation(points, rbins, zmax, box_size, jackknife_nside=0, **kwargs):
    """
    Calculate the projected correlation function wp(rp) and its covariance
    matrix for a periodic box, with the plane-parallel approximation and
    the Jackknife method.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        The last column will be used as the redshift distance.
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    zmax : float
        The integral of \pi goes from -zmax to zmax (redshift distance).
    box_size : float
        The side length of the periodic box.
    jackknife_nside : int, optional (Default: 0)
        If <= 1 , it will not do Jackknife.

    Returns
    -------
    wp : ndarray
        A 1-d array that has wp. The length of this returned array would be
        len(rbins) - 1.
    wp_cov : ndarray (returned if jackknife_nside > 1)
        The len(wp) by len(wp) covariance matrix of wp.
    """

    if 'bias_correction' in kwargs:
        print('`bias_correction` is obsolete. No correction is applied.')

    points, N = _check_points(points)
    rbins = _check_rbins(rbins)
    rbins_sq = rbins*rbins

    max_distances = np.array([rbins[-1], rbins[-1], zmax])
    pairs_rand = get_random_pair_counts(N, box_size, rbins, zmax)
    jackknife_nside = int(jackknife_nside or 0)

    if jackknife_nside <= 1: #no jackknife
        pairs = np.zeros(len(rbins)-1, dtype=np.int)
        for _, _, pos in _get_pairs_max_box(points, points, max_distances, periodic_box_size=box_size):
            pairs += _fast_histogram1d(_reduce_2d_distance_square(pos), rbins_sq)
        return (pairs/pairs_rand - 1.0) * zmax*2.0

    else: #do jackknife
        jack_ids = get_jackknife_ids(points, box_size, jackknife_nside)
        n_jack = jackknife_nside*jackknife_nside
        jack_counts = np.bincount(jack_ids, minlength=n_jack)
        jack_pairs_rand_scale = (N-jack_counts)*(N-jack_counts)/float(N*N)
        del jack_counts

        pairs = np.zeros((n_jack, len(rbins)-1), dtype=np.int)
        auto_pairs = np.zeros_like(pairs)

        for i, j, pos in _get_pairs_max_box(points, points, max_distances, periodic_box_size=box_size):
            jid = jack_ids[i]
            pos = _reduce_2d_distance_square(pos)
            pos_auto = pos[jack_ids[j] == jid]
            pairs[jid] += _fast_histogram1d(pos, rbins_sq)
            auto_pairs[jid] += _fast_histogram1d(pos_auto, rbins_sq)
        del i, j, pos, jack_ids, points

        pairs_sum = pairs.sum(axis=0)
        wp_full = (pairs_sum/pairs_rand - 1.0) * zmax*2.0

        pairs = pairs_sum - pairs*2 + auto_pairs
        wp_jack = (pairs / pairs_rand / jack_pairs_rand_scale[:, np.newaxis] \
                / _jackknife_2d_random(rbins, box_size, jackknife_nside) \
                - 1.0) * zmax*2.0
        wp_cov = np.cov(wp_jack, rowvar=0, bias=1)*(n_jack-1)

        return wp_full, wp_cov


def correlation3d(points, rbins, box_size):
    """
    Calculate the 3D correlation function xi(r) for a periodic box.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns).
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    box_size : float
        The side length of the periodic box.

    Returns
    -------
    xi : ndarray
        A 1-d array that has wp. The length of this returned array would be
        len(rbins) - 1.
    """

    points, N = _check_points(points)
    rbins = _check_rbins(rbins)

    pairs = np.zeros(len(rbins)-1, dtype=np.int)

    for _, _, d in _get_pairs_max_sphere(points, points, rbins[-1], periodic_box_size=box_size):
        pairs += _fast_histogram1d(d, rbins)

    return pairs / get_random_pair_counts(N, box_size, rbins) - 1.0
