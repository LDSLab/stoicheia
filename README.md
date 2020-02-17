# Stoicheia Tensor Storage and Retrieval

# Python API
In most cases you will probably find yourself interacting with the stoicheia APIs through Python, although other bindings will be written in the future. Obviously, you'll also be able to access it through Rust, and hopefully Node.

## Catalog: open a connection
Catalogs are a connection to the datastore for this collection of tensors. By default, this creates an SQLite based catalog if one doesn't already exist, because they are very convenient work with for small installations.
```py
    from stoicheia import Catalog, Patch, Quilt, Axis
    cat = Catalog("example.db")
```

## Patches: labeled slices of tensors
```py
# You can get a slice of any tensor
patch = cat.fetch(
    "tot_sal_amt",  # <- Quilt name
    "latest",       # <- Tag name ("latest" is the default)
    # The rest of the arguments specify the slice you want.

    # You can select by axis labels (not by storage order!)
    itm = [1,2,3],

    # slice() or None both will get the whole axis.
    lct = slice(),

    # Giving just one label will not remove that axis
    # (because that makes merging patches easier)
    day = 721,
)
```

Leaving out any dimensions means to get the whole dimension,
so this reads the whole tensor:
```py
patch = cat.fetch("tot_sal_amt")
```

## Slicing contiguous patches
You can also specify contiguous slices of an axis, but **these are subtle.**

> **Axes have a defined order, but it might not be sorted.**
```py
patch = cat.fetch(
    "tot_sal_amt",
    "latest",
    itm = [1,2,3],
    lct = slice(1001, 1020),
    day = slice(720, 750),
)
```

It may seem like an odd quirk, but the order of an axis is important for [locality of reference](1) and profoundly affects both latency and throughput. This is because patches are stored contiguously whenever possible, and you can configure the storage patterns by changing axis orders.

For the same reason, you can only append to an axis, not permute it, because shuffling it would change what patches store what. So to keep the catalog consistent it would incur possibly massive cascading rebuilds of every patch.



## Commit a patch to the catalog
You can commit a patch to a catalog using a similar method, but the labels will be read from the patch, so you don't need to specify the tensor slice.
```py
cat.commit(
    quilt = "tot_sal_amt",
    tag = "latest", # <- Tags have special meaning in stoicheia
    message = "Elements have been satisfactorily frobnicated",
    patch
)
# "master" and "latest" are the defaults, so you can write:
cat.commit(
    quilt = "tot_sal_amt",
    message = "Elements have been satisfactorily frobnicated",
    patch
)
```
> Tags must be unique: **this will untag any other commits by the same name**

## Untag a patch (to delete it)
Because tensors can be arbitrarily large, you can more easily "delete" commits from stoicheia than from an SCM to manage your storage space. The method is rather simple, you just untag them:
```py
    cat.untag("tot_sal_amt", tag="final_v2_1_test_stage4")
```
**This means you can't access this commit anymore**, and as a result, it can be elided into the child commits, or if it is a leaf, it can be deleted entirely. This proceeds recursively, so that deleting the last of a chain of commits might delete a lot.
    


## Quilts: convenient access to a specific commit
Patches are parts of tensors, and if you deal with the same branch and tag
repeatedly then it may make sense to use a quilt to express that tersely:
```py
# The axis list determines the order you slice and receive patches
quilt = cat.quilt(
    "tot_sal_amt",
    tag = "latest",
    axes = ["itm", "lct", "day"]
)
patch = quilt[[1,2,3], :, 721:]

# __setattr__ doesn't work because you need to create a new commit instead:
# Luckily, this is not hard. You can override the tag if you want.
quilt.commit(
    message = "Engines on",
    patch
)
```

## Using a patch once you have one
The most familiar way to use a patch by exporting it to numpy, which is simple and comes with no strings attached, but it will copy it twice.
```py
# axes:     [(str, np.array(.., dtype=int))]
# content:  np.array(.., dtype=np.float32)
axes, content = patch.export()

# Remove some outliers
content[content > 10000] = content[content <= 10000].mean()

new_patch = Patch.from_content(axes, content)
```

There will probably be better ways to access and mutate the data by brorrowing it in the future, which should make small changes both more efficient and more convenient.

## Creating a patch
You can create a patch in several ways, depending on what data you already have

### From thin air
```py
pat = Patch.from_content(
    {
        "lct": [1, 4, 3, 2],
        "itm": [1001, 1002, 1003, 1004]
    },
    # If you leave out the content, it will be zeros
    content = np.eye(4),
)
```

### From a quilt
It may have escaped your notice that you can read a patch from an area of the quilt that
doesn't exist yet. It does incur a little IO to find if any patches exist but it's convenient.

```py
patch = quilt[[1,2,3], :, 721:]
```

### From another patch
```py
pat = pat.clone()
```


[1]: https://en.wikipedia.org/wiki/Locality_of_reference