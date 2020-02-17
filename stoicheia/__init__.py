class Catalog:
    def __init__(self, _url):
        self._quilts = {} # name -> obj
        self._axes   = {} # name -> array
        self._commits = {} # (quilt_name, name) -> [patch id, ..]
        self._patches = {} # id -> obj

    def __getitem__(self, name):
        ''' Get a quilt or axis by name '''
        if name in quilt;
            return self._quilts[name]
        else if name in axes:
            return self._axes[name]
        else:
            raise KeyError, "No axis or quilt named {}".format(name)

    def create_quilt(self, name, axes):
        ''' Create a new quilt from a name and some axes. '''
        assert name not in self._quilts, "A quilt already exists by the name {}".format(name)
        assert all(axis in self._axes for axis in axes), "All axes must exist before a quilt is created on them."
        return Quilt(self, name, axes)
    
    def fetch(self, quilt, tag=None, **axis_selections):
        assert quilt in self._quilts, "No quilt named {}".format(quilt)
        axes = self._quilts[quilt]._axes
        assert all(axis in self._axes for axis in axes), "All slice axes must be present in the quilt they are selected from: {} contains {} but you selected {}".format(quilt, list(axes), list(axis_selections))
        tag = tag or "master"
        assert (quilt, tag) in self._commits, "No commit tagged {} could be found for quilt {}. Have you committed yet?".format(tag, quilt)
        # This part is already implemented
        return
    

class Quilt:
    ''' Convenience class to make fetching and setting patches more terse '''
    def __init__(self, catalog, name, axes):
        self._catalog = catalog
        self._name = name
        self._axes = axes
    
    def __getitem__(self, *slices):
        return Patch(self._axes, np.array([1]))

    def fetch(self, tag, **axis_selections):
        
    
    def commit(self, patch, tag=None):
        self._patches.append(patch)

