use crate::{Fallible, Label, StoiError};
use std::collections::HashSet;
use std::convert::{From, TryFrom};

/// A sequence of distinct signed integer labels uniquely mapping to indices of an axis
///
///  - In a dense patch, it represents the storage order along one dimension
///  - In a catalog, it determines storage order for all the quilts
///  - If fully sparse patches are supported in the future, axes may then be permitted to repeat
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct Axis {
    pub name: String,
    labels: Vec<Label>,
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
    pub fn range<T: ToString>(name: T, range: std::ops::Range<i64>) -> Axis {
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
    ///     use stoicheia::{Axis, Label};
    ///     let mut left = Axis::new("a", vec![
    ///         Label(1), Label(2), Label(4), Label(5)
    ///     ]).unwrap();
    ///     let right = Axis::new("a", vec![
    ///         Label(1), Label(3), Label(7)
    ///     ]).unwrap();
    ///     left.union(&right);
    ///     assert_eq!(left.labels(), &[
    ///         Label(1), Label(2), Label(4), Label(5), Label(3), Label(7)
    ///     ]);
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
