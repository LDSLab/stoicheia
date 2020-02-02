use crate::{Fallible, Label, StoiError};
use std::collections::HashSet;

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
        }.check_distinct()
    }

    /// Create an empty axis with just a name
    pub fn empty<T: ToString>(name: T) -> Axis {
        Axis {
            name: name.to_string(),
            labels: Vec::new(),
        }
    }

    /// Check if the Axis has no duplicates. O(n) complexity. Useful to check after deserialization.
    /// 
    /// It takes its value because if it isn't distinct it probably shouldn't exist anyway
    pub fn check_distinct(self) -> Fallible<Self> {
        if self.labels.iter().collect::<HashSet<_>>().len() != self.labels.len() {
            Err(StoiError::InvalidValue("Axis labels must not be duplicated"))
        } else {
            Ok(self)
        }
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
    pub fn union(&mut self, other: &Axis) {
        // Hash to speed up duplicate search, and then add only new labels
        let hash: HashSet<_> = self.labels.iter().copied().collect();
        other
            .labels
            .iter()
            .filter(|label| !hash.contains(label))
            .for_each(|label| self.labels.push(*label));
    }
}
