from qamomile.core.converters.utils import is_close_zero

def test_is_close_zero():
    val = 1e-16
    assert is_close_zero(val) 
    val = 1e-14
    assert not is_close_zero(val)
    val = 0
    assert is_close_zero(val)

    val = 1e-14
    assert is_close_zero(val, abs_tol=1e-14)