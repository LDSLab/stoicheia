use crate::Label;
use std::collections::HashSet;

/// A sequence of signed integer labels uniquely mapping to indices of an axis
/// 
/// The meaning and restrictions of the labels depend a lot on the context
/// 
///  - In a dense patch, it represents the storage order along one dimension,
///    and it needs to be unique.
///  - In a sparse patch, it is the coordinates of each populated cell,
///    and it does not need to be unique.
///  - In a catalog, it determines storage order for all the quilts,
///    and it needs to be unique.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct Axis {
    pub name: String,
    pub labels: Vec<Label>,
}
impl Axis {
    /// Create a new named axis with a set of labels
    pub fn new<T: ToString>(name: T, labels: Vec<Label>) -> Axis {
        Axis { name: name.to_string(), labels }
    }

    /// Get the length of the underlying vector. This includes duplicates, if the Axis has any.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Get how many unique labels are in the axis, discarding duplicates. O(n) complexity
    pub fn distinct_len(&self) -> usize {
        self.labels.iter().collect::<HashSet<_>>().len()
    }

    /// Check if the Axis has no duplicates. O(n) complexity
    pub fn is_distinct(&self) -> bool {
        self.distinct_len() == self.len()
    }

    /// Merge the labels of two axes, removing duplicates and appending new elements
    ///
    /// This will not change labels in self, because downstream that means patches would need to
    /// be rebuilt.
    ///
    ///     use stoicheia::{Axis, Label};
    ///     let mut left = Axis::new("a", vec![
    ///         Label(1), Label(2), Label(4), Label(5)
    ///     ]);
    ///     let right = Axis::new("a", vec![
    ///         Label(1), Label(3), Label(7)
    ///     ]);
    ///     left.union(&right);
    ///     assert_eq!(left.labels, vec![
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
