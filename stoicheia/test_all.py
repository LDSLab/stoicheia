from .stoicheia import Catalog

def test_catalog_in_memory():
    cat = Catalog()

def test_catalog_on_disk():
    cat = Catalog("/tmp/foo.db")

def test_create_quilt():
    cat = Catalog()
    cat.create_quilt("sales", ["itm", "lct", "day"])
    cat.fetch("sales", "latest", itm=1, lct=[2,3,4])