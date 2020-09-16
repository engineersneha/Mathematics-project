#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    This file is subject to the terms and conditions defined in
    file 'LICENSE.txt', which is part of this source code package.

    Written by Dr. Gianmarco Mengaldo, May 2020.
'''



def prettify(label, arr):
    '''
    Generates a pretty printed NumPy array with an assignment.
    Optionally, it transposes column vectors so they are drawn
    on one line. Strictly speaking arr can be any time convertible
    by `str(arr)`, but the output may not be what you want if
    the type of the variable is not a scalar or an ndarray.

    Examples
    --------
    >>> pprint('cov', np.array([[4., .1], [.1, 5]]))
    cov = [[4.  0.1]
           [0.1 5. ]]

    >>> print(pretty_str('x', np.array([[1], [2], [3]])))
    x = [[1 2 3]].T
    '''

    def is_column_vector(a):
        '''return true if a is a column vector'''
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except (AttributeError, IndexError):
            return False

    if label is None: label = ''
    if label: label += ' = '
    if is_column_vector(arr): return label + str(arr.T).replace('\n', '') + '.T'
    rows = str(arr).split('\n')
    if not rows: return ''
    s = label + rows[0]
    pad = ' ' * len(label)
    for line in rows[1:]: s = s + '\n' + pad + line
    return s
