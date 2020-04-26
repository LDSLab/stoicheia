use crate::{Fallible, Label, StoiError};
use std::collections::HashSet;
use std::convert::{From, TryFrom};
use std::fmt;

/// A sequence of distinct signed integer labels uniquely mapping to indices of an axis
///
///  - In a dense patch, it represents the storage order along one dimension
///  - In a catalog, it determines storage order for all the quilts
///  - If fully sparse patches are supported in the future, axes may then be permitted to repeat
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct Axis {
    pub name: String,
    labels: Vec<Label>,
}
impl fmt::Debug for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("Axis")
            .field("name", &self.name)
            .field("labels", &&self.labels[..3])
            .finish()?;
        Ok(())
    }
}
impl Axis {
    /// Create a new named axis with a set of labels
    pub fn new<T: ToString>(name: T, labels: Vec<Label>) -> Fallible<Axis> {
        Axis {
            name: name.to_string(),
            labels,
        }
        .check_distinct()
    }

    /// Create a new named axis with a set of labels, assuming they are unique
    pub(crate) fn new_unchecked<T: ToString>(name: T, labels: Vec<Label>) -> Axis {
        Axis {
            name: name.to_string(),
            labels,
        }
    }

    /// Create an empty axis with just a name
    pub fn empty<T: ToString>(name: T) -> Axis {
        Axis {
            name: name.to_string(),
            labels: Vec::new(),
        }
    }

    /// Create an axis from a consecutive range, useful for tests
    pub fn range<T: ToString, R: IntoIterator<Item = i64>>(name: T, range: R) -> Axis {
        Axis {
            name: name.to_string(),
            labels: range.into_iter().collect(),
        }
    }

    /// Check if the Axis has no duplicates. O(n) complexity. Useful to check after deserialization.
    ///
    /// It takes its value because if it isn't distinct it probably shouldn't exist anyway
    pub fn check_distinct(self) -> Fallible<Self> {
        // Switched from HashSet to sorting for a 15-fold best case speedup and about 20% worst-case slowdown
        // see axis benchmarks for details
        let mut l = self.labels().to_vec();
        l.sort_unstable();
        for i in 1..l.len() {
            if l[i - 1] == l[i] {
                return Err(StoiError::InvalidValue(
                    "Axis labels must not be duplicated",
                ));
            }
        }
        Ok(self)
    }

    /// Get a reference to the labels
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Get a reference to the labels, which is O(n) to construct a hashset
    pub fn labelset(&self) -> HashSet<Label> {
        self.labels.iter().copied().collect()
    }

    /// Get the number of selected labels in this axis (in most cases this is not all possible labels)
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Merge the labels of two axes, removing duplicates and appending new elements
    ///
    /// This will not change labels in self, because downstream that means patches would need to
    /// be rebuilt.
    ///
    ///     use stoicheia::Axis;
    ///     let mut left = Axis::new("a", vec![1, 2, 4, 5]).unwrap();
    ///     let right = Axis::new("a", vec![1, 3, 7]).unwrap();
    ///     left.union(&right);
    ///     assert_eq!(left.labels(), &[1, 2, 4, 5, 3, 7]);
    ///
    /// Returns true iff self was actually mutated in the process
    pub fn union(&mut self, other: &Axis) -> bool {
        // Hash to speed up duplicate search, and then add only new labels
        let hash: HashSet<_> = self.labels.iter().copied().collect();
        let mut mutated = false;
        other
            .labels
            .iter()
            .filter(|label| !hash.contains(label))
            .inspect(|_| {
                mutated = true;
            })
            .for_each(|label| self.labels.push(*label));
        mutated
    }

    /// Find the smallest aligned power-of-two block enclosing an interval.
    /// 
    /// Accepts:
    ///     start: an index (not label) included in the interval
    ///     end_inclusive: an index (not label) at the end of the interval
    /// 
    /// Returns:
    ///     (S, E) where:
    ///         -   S is the beginning of a block, inclusive
    ///         -   E is the end of a block, inclusive
    ///         -   E-S is a power of two
    ///         -   S is divisible by E-S
    ///         -   S <= start <= end_exclusive <= E
    pub(crate) fn get_block(start: u64, end_inclusive: u64) -> (u64, u64) {
        if start == end_inclusive {
            (start, start)
        } else {
            let prefix_len = (start ^ end_inclusive).leading_zeros();
            let prefix_mask = u64::max_value() >> prefix_len;
            (start & ! prefix_mask, start | prefix_mask)
        }
    }
}

impl From<&Axis> for Axis {
    fn from(a: &Axis) -> Self {
        a.clone()
    }
}

impl<L: IntoIterator<Item = Label>> TryFrom<(&str, L)> for Axis {
    type Error = StoiError;
    fn try_from(x: (&str, L)) -> Result<Self, StoiError> {
        Axis::new(x.0, x.1.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Axis, Catalog, Label, StorageTransaction};

    #[test]
    fn test_create_axis() {
        let mut cat = Catalog::connect("").unwrap();
        let mut txn = cat.begin().unwrap();

        let ax = txn
            .get_axis("xjhdsa")
            .expect("Getting an empty axis should be fine");
        assert!(ax.labels() == &[] as &[Label]);

        let ref ax = Axis::new("uyiuyoiuy", vec![1, 5]).expect("Should be able to create an axis");

        // Union an axis
        txn.union_axis(&ax)
            .expect("Should be able to create an axis by union");
        let ax = txn
            .get_axis("uyiuyoiuy")
            .expect("Axis should exist after union")
            .to_owned();
        // Note how we had to clone the axis. This is interesting, and for good reason!
        // Unioning an axis would cause any existing axes to possibly be invalidated.
        // So you can't union while there are still any references to Axes.
        // Your axes can only be out of date if you make a copy. And in this case, we
        // have chosen to do that.
        assert_eq!(ax.labels(), &[1, 5]);

        txn.union_axis(&ax).expect("Union twice is a no-op");
        let ax = txn
            .get_axis("uyiuyoiuy")
            .expect("Axis should still exist after second union");
        assert_eq!(ax.labels(), &[1, 5]);

        txn.union_axis(&Axis::new("uyiuyoiuy", vec![0, 5]).unwrap())
            .expect("Union should append");
        let ax = txn.get_axis("uyiuyoiuy").unwrap();
        assert_eq!(ax.labels(), &[1, 5, 0]);
    }

    #[test]
    fn test_split_patch() {
        assert_eq!(Axis::get_block(8, 10), (8, 11));
        assert_eq!(Axis::get_block(7, 10), (0, 15));
        assert_eq!(Axis::get_block(6, 9), (0, 15));
        assert_eq!(Axis::get_block(10, 10), (10, 10));
        assert_eq!(Axis::get_block(0, 0), (0, 0));
    }
}