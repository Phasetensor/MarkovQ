from markovq.classification import (
    accessible,
    closed_classes,
    communicates,
    communicating_classes,
    is_irreducible,
    state_period,
)


def test_transient_bridge():
    p = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
    assert accessible(p, 0, 2)
    assert accessible(p, 0, 0)
    assert not communicates(p, 0, 2)
    assert communicates(p, 1, 2)
    assert set(communicating_classes(p)) == {(0,), (1, 2)}
    assert closed_classes(p) == [(1, 2)]
    assert state_period(p, 0) == 0
    assert state_period(p, 1) == 2
    assert not is_irreducible(p)
