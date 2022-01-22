use ndarray::Array1;
use num_traits::Float;

/// Preference is the value representing the degree to which a data point will act as its own exemplar,
/// with lower (more negative) values yielding fewer clusters.
///
/// - Median: Use median similarity value as preference
/// - List: Use provided preference list
/// - Value: Assign all members the same preference value
#[derive(Debug, Clone)]
pub enum Preference<'a, F>
where
    F: Float + Send + Sync,
{
    Median,
    List(&'a Array1<F>),
    Value(F),
}
