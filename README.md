# Stoicheia Tensor Storage and Retrieval

# Python API
In most cases you will probably find yourself interacting with the stoicheia APIs through Python, although other bindings will be written in the future. Obviously, you'll also be able to access it through Rust, and hopefully Node.

## Catalog: open a connection
Catalogs are a connection to the datastore for this collection of tensors. By default, this creates an SQLite based catalog if one doesn't already exist, because they are very convenient work with for small installations.
```py
    from stoicheia import Catalog, Patch, Quilt, Axis
    disk_db = Catalog("example.db") # Connect and create a new database if necessary
    mem_db  = Catalog("") # Use an in-memory database
    mem_db2 = Catalog("") # References the same database as mem_db (not a new one)
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

    # Omitting an axis or giving None will get the whole axis.
    lct = None,

    # Giving just one label will not remove that axis
    # (because that makes merging patches easier)
    day = 721,
)
# Or you can get the whole tensor.
# There are memory guardrails so this may fail if you request something huge.
patch = cat.fetch("tot_sal_amt")
```

## Slicing contiguous patches
You can also specify contiguous slices of an axis, by giving the first and last elements.

```py
patch = cat.fetch(
    "tot_sal_amt",
    "latest",
    itm = [1,2,3],
    lct = (1001, 1020),
    day = (720, 750),
)
```
There are three important parts that may surprise you:
1. **Axes have a defined order, but it might not be sorted.**
   - You define the axis' order when you create it, and that order informs splitting and searching patches.
   - **Adjacent labels imply adjacent elements** in all quilts with those axes.
   - Axis order affects [locality of reference](1), and profoundly affects latency and throughput. So when you define axes, be sure to order it so labels used together are adjacent to each other.
2. **Ranges are inclusive on both ends** because the axes are not sorted.
   - If you wanted a half-open interval you can drop the last element. (We did this because if you had half-open and wanted a closed interval, you'd have to find the next element because it's not merely `n+1`.
3. **Axes are append-only and unique, so they only support `union()`**
   - Axes have to be unique because we use them like primary keys.
   - Permuting axes would cause every affected patch to be rebuilt to maintain storage order. But you can still copy all the quilts onto the new axis instead, which may be faster.


## Commit a patch to the catalog
You can commit a patch to a catalog using a similar method, but the labels will be read from the patch, so you don't need to specify the tensor slice.
```py
cat.commit(
    quilt = "tot_sal_amt",
    parent_tag = "latent", # <- Apply the changes to this tag
    new_tag = "latest",    # <- Name the result (can be the same)
    message = "Elements have been satisfactorily frobnicated",
    patch
)
# "latest" is the default tag, so you can write:
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
    new_tag = "latest",
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