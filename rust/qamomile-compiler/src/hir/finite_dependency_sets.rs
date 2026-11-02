//! Bounded finite-set propagation over a dependency graph.
//!
//! Higher-order HIR normalization uses this solver for distinct candidate
//! lattices while retaining each pass's own values, limits, and diagnostics.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
};

pub(super) struct FiniteDependencyFacts<K, V> {
    pub(super) direct: HashSet<V>,
    pub(super) dependencies: HashSet<K>,
}

impl<K, V> Default for FiniteDependencyFacts<K, V> {
    fn default() -> Self {
        Self {
            direct: HashSet::new(),
            dependencies: HashSet::new(),
        }
    }
}

pub(super) fn solve_finite_dependency_sets<K, V, E>(
    facts: &HashMap<K, FiniteDependencyFacts<K, V>>,
    mut visit: impl FnMut(),
    mut validate_cardinality: impl FnMut(usize) -> Result<(), E>,
) -> Result<HashMap<K, HashSet<V>>, E>
where
    K: Clone + Eq + Hash,
    V: Clone + Eq + Hash,
{
    let mut values = facts
        .iter()
        .map(|(key, facts)| (key.clone(), facts.direct.clone()))
        .collect::<HashMap<_, _>>();
    let mut dependents = HashMap::<K, Vec<K>>::new();
    for (target, facts) in facts {
        for dependency in &facts.dependencies {
            dependents
                .entry(dependency.clone())
                .or_default()
                .push(target.clone());
        }
    }

    let mut queue = facts
        .iter()
        .map(|(key, facts)| {
            (
                key.clone(),
                facts.direct.iter().cloned().collect::<Vec<_>>(),
            )
        })
        .collect::<VecDeque<_>>();
    while let Some((source, additions)) = queue.pop_front() {
        visit();
        for target in dependents.get(&source).into_iter().flatten() {
            let target_values = values.entry(target.clone()).or_default();
            let mut propagated = Vec::new();
            for value in &additions {
                if target_values.insert(value.clone()) {
                    propagated.push(value.clone());
                }
            }
            validate_cardinality(target_values.len())?;
            if !propagated.is_empty() {
                queue.push_back((target.clone(), propagated));
            }
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{FiniteDependencyFacts, solve_finite_dependency_sets};
    use std::collections::{HashMap, HashSet};

    fn facts(
        direct: impl IntoIterator<Item = u8>,
        dependencies: impl IntoIterator<Item = &'static str>,
    ) -> FiniteDependencyFacts<&'static str, u8> {
        FiniteDependencyFacts {
            direct: direct.into_iter().collect(),
            dependencies: dependencies.into_iter().collect(),
        }
    }

    #[test]
    fn closes_transitive_diamond_and_cycle_dependencies() {
        let graph = HashMap::from([
            ("root", facts([1], [])),
            ("left", facts([2], ["root", "right"])),
            ("right", facts([], ["root", "left"])),
            ("join", facts([3], ["left", "right"])),
        ]);
        let mut visits = 0;
        let values = solve_finite_dependency_sets(
            &graph,
            || visits += 1,
            |count| (count <= 3).then_some(()).ok_or(count),
        )
        .expect("bounded closure");

        assert_eq!(values["root"], HashSet::from([1]));
        assert_eq!(values["left"], HashSet::from([1, 2]));
        assert_eq!(values["right"], HashSet::from([1, 2]));
        assert_eq!(values["join"], HashSet::from([1, 2, 3]));
        assert!(visits >= graph.len());
    }

    #[test]
    fn rejects_growth_before_returning_a_partial_solution() {
        let graph = HashMap::from([
            ("source", facts([1, 2], [])),
            ("target", facts([3], ["source"])),
        ]);

        assert_eq!(
            solve_finite_dependency_sets(
                &graph,
                || {},
                |count| { (count <= 2).then_some(()).ok_or(count) }
            ),
            Err(3)
        );
    }
}
