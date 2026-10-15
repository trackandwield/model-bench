from bench import dataset


def test_battlecard_cases_are_valid():
    cases = dataset.load("battlecard")
    assert len(cases) == 20
    assert dataset.validate(cases) == []


def test_battlecard_has_tricky_cases():
    counts = dataset.summary(dataset.load("battlecard"))
    assert counts["difficulty"]["tricky"] >= 3
