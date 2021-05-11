# -*- coding: UTF-8 -*-

from emissions.data import load_data, clean_data

def test_load_data():
    df = load_data()
    assert df.shape == (187503, 10)
    assert df.RESULT.nunique() == 2

def test_clean_data():
    tmp = load_data()
    df = clean_data(tmp)
    assert df.shape == (174424, 8)
    assert df.RESULT.nunique() == 2
    assert df.VEHICLE_TYPE.nunique() == 7
    assert df.VEHICLE_AGE.nunique() == 39
    assert df.MILE_YEAR.max() < 100000
    assert df.GVWR.min() == 847
    assert df.GVWR.max() == 10000
    assert df.ENGINE_SIZE.min() == 500
    assert df.TEST_TYPE.nunique() == 2
    assert df.TRANS_TYPE.nunique() == 2