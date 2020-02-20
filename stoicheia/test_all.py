from .stoicheia import Catalog
import numpy as np

def test_catalog_in_memory():
    cat = Catalog()

def test_catalog_on_disk():
    cat = Catalog("/tmp/foo.db")

def test_create_quilt():
    cat = Catalog()
    cat.create_quilt("sales", ["itm", "lct", "day"])
    pat = cat.fetch("sales", "latest", itm=1, lct=[2,3,4])
    axes, content = pat.export()
    assert np.array_equal(axes[0], np.array([1]))
    assert np.array_equal(axes[1], np.array([2,3,4]))
    assert np.array_equal(content, np.array([[0,0,0]]))
    