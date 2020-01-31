use crate::Label;
use std::collections::HashSet;

/// A sequence of nonsequential, distinct, signed integer labels of an tensor's axis
/// 
/// In a patch, it represents the storage order along an axis.
/// In a catalog, it determines storage order for all the quilts.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct Axis {
    pub name: String,
    pub labels: Vec<Label>,
}
impl Axis {
    pub fn new<T: ToString>(name: T, labels: Vec<Label>) -> Axis {
        Axis { name: name.to_string(), labels }
    }
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
