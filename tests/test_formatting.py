from itertools import product

import numpy as np
import pandas as pd
import pytest

from risk_distributions.formatting import cast_to_series, is_computable_empty, make_nan_data


valid_inputs = (np.array([1]), pd.Series([1]), [1], (1,), 1)


@pytest.mark.parametrize('mean, sd', product(valid_inputs, valid_inputs))
def test_cast_to_series_single_ints(mean, sd):
    expected_mean, expected_sd = pd.Series([1]), pd.Series([1])
    out_mean, out_sd = cast_to_series(mean, sd)
    assert expected_mean.equals(out_mean)
    assert expected_sd.equals(out_sd)


valid_inputs = (np.array([1.]), pd.Series([1.]), [1.], (1.,), 1.)


@pytest.mark.parametrize('mean, sd', product(valid_inputs, valid_inputs))
def test_cast_to_series_single_floats(mean, sd):
    expected_mean, expected_sd = pd.Series([1.]), pd.Series([1.])
    out_mean, out_sd = cast_to_series(mean, sd)
    assert expected_mean.equals(out_mean)
    assert expected_sd.equals(out_sd)


valid_inputs = (np.array([1, 2, 3]), pd.Series([1, 2, 3]), [1, 2, 3], (1, 2, 3))


@pytest.mark.parametrize('mean, sd', product(valid_inputs, valid_inputs))
def test_cast_to_series_array_like(mean, sd):
    expected_mean, expected_sd = pd.Series([1, 2, 3]), pd.Series([1, 2, 3])
    out_mean, out_sd = cast_to_series(mean, sd)
    assert expected_mean.equals(out_mean)
    assert expected_sd.equals(out_sd)


reference = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
valid_inputs = (np.array([1, 2, 3]), reference, [1, 2, 3], (1, 2, 3))


@pytest.mark.parametrize('reference, other', product([reference], valid_inputs))
def test_cast_to_series_indexed(reference, other):
    out_mean, out_sd = cast_to_series(reference, other)
    assert reference.equals(out_mean)
    assert reference.equals(out_sd)

    out_mean, out_sd = cast_to_series(other, reference)
    assert reference.equals(out_mean)
    assert reference.equals(out_sd)


null_inputs = (np.array([]), pd.Series([]), [], ())


@pytest.mark.parametrize('val, null', product([1], null_inputs))
def test_cast_to_series_nulls(val, null):
    with pytest.raises(ValueError, match='Empty data structure'):
        cast_to_series(val, null)

    with pytest.raises(ValueError, match='Empty data structure'):
        cast_to_series(null, val)


def test_cast_to_series_mismatched_index():
    reference = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    other = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match='identically indexed'):
        cast_to_series(reference, other)

    with pytest.raises(ValueError, match='identically indexed'):
        cast_to_series(other, reference)


reference = (np.array([1, 2, 3]), pd.Series([1, 2, 3]), [1, 2, 3], (1, 2, 3))
invalid = (np.array([1]), pd.Series([1]), [1], (1,), 1, 1.,
           np.arange(5), pd.Series(np.arange(5)), list(range(5)), tuple(range(5)))


@pytest.mark.parametrize('reference, other', product(reference, invalid))
def test_cast_to_series_mismatched_length(reference, other):
    with pytest.raises(ValueError, match='same number of values'):
        cast_to_series(reference, other)

    with pytest.raises(ValueError, match='same number of values'):
        cast_to_series(other, reference)


valid_inputs = (np.array([1, 2, 3]), pd.Series([1, 2, 3]), [1, 2, 3], (1, 2, 3), {'x': [1, 2, 3]},
                pd.DataFrame({'x': [1, 2, 3]}), np.array([0, 1, 2, 3, np.nan]), pd.Series([1, 2, 3, np.nan, 0]),
                [0, 1, 2, 3, np.nan], (0, 1, 2, 3, np.nan), {'x': [1, 2, 3], 'y': [1, 0, np.nan]},
                pd.DataFrame([1, 2, 3, np.nan]))


@pytest.mark.parametrize('parameter', valid_inputs)
def test_verify_parameters_valid(parameter):
    assert not is_computable_empty(parameter)


invalid_inputs = (np.array([0, 0, 0]), pd.Series([0, 0, 0]), [0, 0, 0], (0, 0, 0), {'x': [0, 0, 0]},
                  pd.DataFrame({'x': [0, 0, 0]}), np.array([np.nan, np.nan, 0]), pd.Series([np.nan, np.nan, 0]),
                  [np.nan, np.nan, 0], (np.nan, np.nan, 0), {'x': [np.nan, np.nan, 0]},
                  pd.DataFrame({'x': [np.nan, np.nan, 0]}), np.array([np.nan]), pd.Series([np.nan]), [np.nan],
                  {'x': [np.nan]}, pd.DataFrame({'x': [np.nan]}))


@pytest.mark.parametrize('parameter', invalid_inputs)
def test_verify_parameters_invalid(parameter):
    assert is_computable_empty(parameter)


@pytest.mark.parametrize('parameter', invalid_inputs)
def test_format_data_invalid_parameters(parameter):
    if isinstance(parameter, dict):
        data = list(make_nan_data(parameter).values())
    else:
        data = make_nan_data(parameter)
    assert np.all(np.isnan(data))
