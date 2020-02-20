from .stoicheia import Axis, Patch, Catalog
import numpy as np

def test_catalog_in_memory():
    cat = Catalog()

def test_catalog_on_disk():
    cat = Catalog("/tmp/foo.db")

def test_create_quilt():
    cat = Catalog()
    cat.create_quilt("sales", ["itm", "lct", "day"])

def test_fetch_empty():
    cat = Catalog()
    cat.create_quilt("sales", ["itm", "lct", "day"])
    pat = cat.fetch("sales", "latest", itm=1, lct=[2,3,4])
    axes, content = pat.export()
    assert np.array_equal(axes[0], np.array([1]))
    assert np.array_equal(axes[1], np.array([2,3,4]))
    assert np.array_equal(content, np.array([[0,0,0]]))


def test_init_an_axis():
    # Should work fine with a numpy array
    a = Axis("thing", np.array([1,2,3]))
    assert np.array_equal(a.labels(), np.array([1,2,3]))
    assert a.name() == "thing"
    # TODO: Should work fine with a list too
    # a = Axis("thing", [1,2,3])
    # assert np.array_equal(a.labels(), np.array([1,2,3]))

def test_create_patch():
    pat = Patch(
        axes = [
            Axis("itm", np.array([1])),
            Axis("lct", np.array([2,3,4]))
        ],
        content = np.array([[1.0, 2.0, 6.0]], dtype=np.float32)
    )

    axes, content = pat.export()
    assert np.array_equal(axes[0], np.array([1]))
    assert np.array_equal(axes[1], np.array([2,3,4]))
    assert np.array_equal(content, np.array([[1,2,6]]))

# def test_commit_patch():
#     cat = Catalog()
#     cat.create_quilt("sales", ["itm", "lct", "day"])

#     # Same as test_commit_patch
#     pat = Patch(
#         axes = [
#             Axis("itm", [1]),
#             Axis("lct", [2,3,4])
#         ],
#         content = np.array([1,2,6])
#     )

#     axes, content = pat.export()
#     assert np.array_equal(axes[0], np.array([1]))
#     assert np.array_equal(axes[1], np.array([2,3,4]))
#     assert np.array_equal(content, np.array([[1,2,6]]))
    