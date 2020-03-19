from ipfx.stimulus import StimulusOntology
import pytest


@pytest.fixture()
def ontology():
    return StimulusOntology([[('name', 'long square'),
                              ('code', 'LS')],
                             [('name', 'noise', 'noise 1'),
                              ('code', 'C1NS1')],
                             [('name', 'noise', 'noise 2'),
                              ('code', 'C1NS2')]])


def test_find(ontology):
    stims = ontology.find('C1NS1')

    stims = ontology.find('noise')
    assert len(stims) == 2


def test_find_one(ontology):
    stim = ontology.find_one('LS')

    assert stim.tags(tag_type='name')[0][-1] == 'long square'



def test_has(ontology):
    assert ontology.stimulus_has_any_tags('C1NS1', ('noise',))
    assert ontology.stimulus_has_any_tags('C1NS1', ('noise', 'noise 2'))
    assert not ontology.stimulus_has_all_tags('C1NS1', ('noise', 'noise 2'))
