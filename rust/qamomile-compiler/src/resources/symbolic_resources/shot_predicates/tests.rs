use super::super::SymbolicCount;
use super::*;

fn context() -> (ExpressionArena, CurrentShot, ShotPredicates) {
    let mut arena = ExpressionArena::default();
    let shot = CurrentShot(arena.current_shot().expect("current shot"));
    (arena, shot, ShotPredicates::default())
}

fn recognize(
    proofs: &mut ShotPredicates,
    shot: CurrentShot,
    predicate: ExprId,
    arena: &mut ExpressionArena,
) -> ExprId {
    proofs
        .recognize(predicate, shot, arena)
        .expect("recognition")
        .expect("certified predicate")
}

fn project(
    proofs: &mut ShotPredicates,
    mut count: SymbolicCount,
    arena: &mut ExpressionArena,
) -> SymbolicCount {
    let mut projection = Projection::new(proofs, arena.nodes.len());
    projection.count(&mut count, arena).expect("projection");
    count
}

fn optional(predicate: ExprId, expression: ExprId, arena: &mut ExpressionArena) -> SymbolicCount {
    let value = SymbolicCount::exact_value(expression);
    let zero = SymbolicCount::exact_value(arena.literal_u128(0).expect("zero"));
    SymbolicCount::conditional(&value, &zero, predicate, arena).expect("conditional count")
}

#[test]
fn current_shot_identity_survives_every_symbol_collision() {
    let (mut arena, shot, mut proofs) = context();
    let before = arena.node(shot.0).expect("shot declaration").clone();
    let unrelated = arena
        .intern(before.clone())
        .expect("same full declaration from outside");
    assert_ne!(shot.0, unrelated);
    assert_eq!(arena.current_shot().expect("repeated read"), shot.0);
    assert_eq!(arena.node(unrelated), Some(&before));
    assert_ne!(arena.node(shot.0), Some(&before));
    let moved = arena.node(shot.0).expect("moved declaration").clone();
    let another = arena.intern(moved).expect("second collision");
    assert_ne!(shot.0, another);
    assert_ne!(unrelated, another);
    let eq = arena
        .predicate_comparison(Predicate::Equal, [shot.0, unrelated])
        .expect("comparison");
    assert!(
        proofs
            .recognize(eq, shot, &mut arena)
            .expect("recognition")
            .is_none()
    );
    let declarations = arena.projected_nodes();
    let identities = [shot.0, unrelated, another].map(|id| {
        let symbol = declarations[id.index().expect("index")]
            .symbol()
            .expect("symbol");
        (symbol.scope().to_owned(), symbol.name().to_owned())
    });
    assert_eq!(
        identities
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(identities.iter().all(|(_, name)| name == "shot"));
    assert_eq!(
        arena.ids.get(arena.node(shot.0).expect("shot")),
        Some(&shot.0)
    );
}

#[test]
fn preexisting_source_and_rule_scopes_do_not_capture_shot() {
    let mut arena = ExpressionArena::default();
    for index in 0..10 {
        arena
            .symbol(
                &format!("qamomile.current_shot.{index}"),
                "shot",
                ResourceDomain::Natural,
                "current_shot",
                &[],
            )
            .expect("external declaration");
    }
    let shot = arena.current_shot().expect("private shot");
    assert!(
        matches!(arena.node(shot), Some(Node::Symbol { scope, .. }) if scope == "qamomile.current_shot.10")
    );
}

#[test]
fn positive_integer_modulo_is_natural_and_range_keeps_it() {
    let (mut arena, shot, _) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    for divisor in [-2, 0, 2] {
        let divisor_expression = arena.literal_i64(divisor).expect("divisor");
        let modulo = arena
            .contracted(
                "qamomile.modulo",
                ResourceDomain::Integer,
                [shot.0, divisor_expression],
            )
            .expect("modulo");
        assert_eq!(
            arena.node(modulo).expect("modulo node").domain(),
            if divisor > 0 {
                ResourceDomain::Natural
            } else {
                ResourceDomain::Integer
            }
        );
        let trip = arena
            .contracted(
                "qamomile.range_trip_count",
                ResourceDomain::Natural,
                [zero, modulo, one],
            )
            .expect("range");
        assert_eq!(trip == modulo, divisor > 0);
    }
    let real = arena
        .symbol("user", "r", ResourceDomain::Real, "param", &[])
        .expect("real");
    let modulo = arena
        .contracted("qamomile.modulo", ResourceDomain::Real, [real, one])
        .expect("real modulo");
    assert_eq!(
        arena.node(modulo).expect("node").domain(),
        ResourceDomain::Real
    );
}

#[test]
fn canonical_complements_have_exact_total_and_marginal_bounds() {
    for (first, second) in [
        (Predicate::Equal, Predicate::NotEqual),
        (Predicate::NotEqual, Predicate::Equal),
        (Predicate::Less, Predicate::GreaterEqual),
        (Predicate::LessEqual, Predicate::Greater),
    ] {
        let (mut arena, shot, mut proofs) = context();
        let one = arena.literal_u128(1).expect("one");
        let first = arena
            .predicate_comparison(first, [shot.0, one])
            .expect("first comparison");
        let second = arena
            .predicate_comparison(second, [shot.0, one])
            .expect("second comparison");
        let first = recognize(&mut proofs, shot, first, &mut arena);
        let second = recognize(&mut proofs, shot, second, &mut arena);
        assert_eq!(proofs.atoms.len(), 1);
        let left = optional(first, one, &mut arena);
        let right = optional(second, one, &mut arena);
        let sum = left.then(&right, &mut arena).expect("sequential sum");
        let total = project(&mut proofs, sum, &mut arena);
        assert!(total.exact);
        assert!(arena.is_literal(total.lower, "1"));
        assert_eq!(total.upper, Some(total.lower));
        let marginal = project(&mut proofs, left, &mut arena);
        assert!(!marginal.exact);
        assert!(
            matches!(arena.node(marginal.lower), Some(Node::Minimum { arguments, .. }) if arguments.iter().any(|id| arena.is_literal(*id, "0")) && arguments.iter().any(|id| arena.is_literal(*id, "1")))
        );
    }
}

#[test]
fn reversed_equality_boolean_connectives_and_repeated_reads_are_canonical() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let eq = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("equality");
    let reversed = arena
        .predicate_comparison(Predicate::Equal, [one, shot.0])
        .expect("reversed");
    assert_eq!(
        recognize(&mut proofs, shot, eq, &mut arena),
        recognize(&mut proofs, shot, reversed, &mut arena)
    );
    let same = arena
        .predicate_comparison(Predicate::Equal, [shot.0, shot.0])
        .expect("same context");
    let same = recognize(&mut proofs, shot, same, &mut arena);
    assert_eq!(arena.predicate_value(same), Some(true));
    let lower = arena
        .predicate_comparison(Predicate::LessEqual, [zero, shot.0])
        .expect("lower");
    let both = arena
        .predicate_connective(Predicate::And, [eq, lower])
        .expect("conjunction");
    let not = arena
        .predicate_connective(Predicate::Not, [both])
        .expect("negation");
    let either = arena
        .predicate_connective(Predicate::Or, [both, not])
        .expect("disjunction");
    let either = recognize(&mut proofs, shot, either, &mut arena);
    let count = optional(either, one, &mut arena);
    let result = project(&mut proofs, count, &mut arena);
    assert!(result.exact && arena.is_literal(result.lower, "1"));
}

#[test]
fn different_atoms_and_repeated_predicates_keep_enclosures() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let first = arena
        .predicate_comparison(Predicate::Equal, [shot.0, zero])
        .expect("first");
    let second = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("second");
    let first = recognize(&mut proofs, shot, first, &mut arena);
    let second = recognize(&mut proofs, shot, second, &mut arena);
    let left = optional(first, one, &mut arena);
    for predicate in [first, second] {
        let right = optional(predicate, one, &mut arena);
        let sum = left.then(&right, &mut arena).expect("sum");
        let result = project(&mut proofs, sum, &mut arena);
        assert!(!result.exact);
        assert!(
            matches!(arena.node(result.upper.expect("upper")), Some(Node::Maximum { arguments, .. }) if arguments.iter().any(|id| arena.is_literal(*id, "2")))
        );
    }
}

#[test]
fn unrelated_symbols_and_partial_integer_predicates_are_not_certified() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let param = arena
        .symbol("sample", "shot", ResourceDomain::Integer, "param", &[])
        .expect("source parameter");
    for operand in [
        param,
        arena
            .contracted("qamomile.modulo", ResourceDomain::Integer, [shot.0, zero])
            .expect("partial modulo"),
    ] {
        let condition = arena
            .predicate_comparison(Predicate::Equal, [operand, zero])
            .expect("comparison");
        assert!(
            proofs
                .recognize(condition, shot, &mut arena)
                .expect("recognition")
                .is_none()
        );
    }
}

#[test]
fn guard_removal_never_exposes_partial_cost_arithmetic() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let nonzero = arena
        .predicate_comparison(Predicate::NotEqual, [shot.0, zero])
        .expect("nonzero");
    let nonzero = recognize(&mut proofs, shot, nonzero, &mut arena);
    let division = arena
        .contracted(
            "qamomile.floor_divide",
            ResourceDomain::Integer,
            [one, shot.0],
        )
        .expect("partial division");
    let count = optional(nonzero, division, &mut arena);
    let result = project(&mut proofs, count, &mut arena);
    assert!(!result.exact && result.upper.is_none());
    assert_eq!(result.unknown.as_ref(), &[TOTALITY_REASON]);
    let valid = optional(nonzero, one, &mut arena);
    let invalid = arena
        .contracted(
            "qamomile.floor_divide",
            ResourceDomain::Integer,
            [one, zero],
        )
        .expect("invalid unused branch");
    let false_guard = arena.predicate_literal(false).expect("false");
    proofs
        .predicates
        .insert(false_guard, Truth::constant(false));
    let guarded = arena
        .intern(Node::Piecewise {
            domain: ResourceDomain::Natural,
            arguments: Box::new([invalid, valid.lower]),
            predicates: Box::new([false_guard]),
        })
        .expect("lazy piecewise");
    let result = project(&mut proofs, SymbolicCount::exact_value(guarded), &mut arena);
    assert!(result.upper.is_some() && result.unknown.is_empty());
}

#[test]
fn metric_domains_provenance_and_unproved_equality_survive_projection() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, zero])
        .expect("comparison");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let half = arena
        .intern(Node::RationalLiteral {
            numerator: "1".into(),
            denominator: "2".into(),
            domain: ResourceDomain::Rational,
        })
        .expect("half");
    let mut count = optional(predicate, half, &mut arena);
    assert_eq!(
        arena.node(count.lower).expect("conditioned cost").domain(),
        ResourceDomain::Rational
    );
    count.rule_identities = Box::new(["rule-a".into()]);
    count.unknown = Box::new(["existing unknown".into()]);
    count.exact = false;
    let result = project(&mut proofs, count, &mut arena);
    assert!(!result.exact);
    assert_eq!(result.rule_identities.as_ref(), &["rule-a"]);
    assert_eq!(result.unknown.as_ref(), &["existing unknown"]);
    assert_eq!(
        arena.node(result.lower).expect("projected cost").domain(),
        ResourceDomain::Rational
    );
}

#[test]
fn proof_budgets_are_shared_and_preserve_unknown_provenance() {
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("comparison");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let mut count = optional(predicate, one, &mut arena);
    count.rule_identities = Box::new(["rule-a".into()]);
    count.unknown = Box::new(["prior unknown".into()]);
    proofs.budget.storage = 0;
    let result = project(&mut proofs, count, &mut arena);
    assert!(!result.exact && result.upper.is_none());
    assert!(result.unknown.iter().any(|reason| reason == BUDGET_REASON));
    assert!(
        result
            .unknown
            .iter()
            .any(|reason| reason == "prior unknown")
    );
    assert_eq!(result.rule_identities.as_ref(), &["rule-a"]);
    let unchanged = project(&mut proofs, SymbolicCount::exact_value(shot.0), &mut arena);
    assert!(unchanged.exact && unchanged.lower == shot.0);
}

#[test]
fn failed_proofs_keep_signed_metric_lower_bounds_guarded() {
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let negative = arena.literal_i64(-2).expect("negative metric contribution");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, zero])
        .expect("condition");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let count = optional(predicate, negative, &mut arena);
    let guarded_lower = count.lower;
    proofs.budget.work = 0;
    let result = project(&mut proofs, count, &mut arena);
    assert_eq!(result.lower, guarded_lower);
    assert!(!result.exact && result.upper.is_none());
    assert_eq!(result.unknown.as_ref(), &[BUDGET_REASON]);
}

#[test]
fn atom_budget_rejects_the_ninth_choice_and_deep_graphs_are_iterative() {
    let (mut arena, shot, mut proofs) = context();
    for index in 0..=MAX_ATOMS {
        let literal = arena.literal_u128(index as u128).expect("literal");
        let predicate = arena
            .predicate_comparison(Predicate::Equal, [shot.0, literal])
            .expect("comparison");
        assert_eq!(
            proofs
                .recognize(predicate, shot, &mut arena)
                .expect("recognition")
                .is_some(),
            index < MAX_ATOMS
        );
    }
    assert_eq!(proofs.atoms.len(), MAX_ATOMS);
    let mut deep = shot.0;
    for _ in 0..20_000 {
        deep = arena.floor(deep).expect("deep expression");
    }
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [deep, shot.0])
        .expect("comparison");
    assert!(
        proofs
            .recognize(predicate, shot, &mut arena)
            .expect("bounded recognition")
            .is_none()
    );
}

#[test]
fn loop_composition_keeps_the_same_shot_decision_and_symbolic_factor() {
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("comparison");
    let opposite = arena
        .predicate_connective(Predicate::Not, [predicate])
        .expect("negation");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let opposite = recognize(&mut proofs, shot, opposite, &mut arena);
    let binder = arena
        .symbol("loop", "i", ResourceDomain::Natural, "loop", &[])
        .expect("binder");
    let left = optional(predicate, one, &mut arena)
        .repeat(shot.0, binder, &mut arena)
        .expect("repeat");
    let right = optional(opposite, one, &mut arena)
        .repeat(shot.0, binder, &mut arena)
        .expect("repeat");
    let sum = left.then(&right, &mut arena).expect("sum");
    let result = project(&mut proofs, sum, &mut arena);
    assert!(result.exact);
    assert_eq!(result.lower, shot.0);
}

#[test]
fn error_bound_composition_is_not_upgraded_to_an_equality() {
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("comparison");
    let opposite = arena
        .predicate_connective(Predicate::Not, [predicate])
        .expect("negation");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let opposite = recognize(&mut proofs, shot, opposite, &mut arena);
    let half = arena
        .intern(Node::RationalLiteral {
            numerator: "1".into(),
            denominator: "2".into(),
            domain: ResourceDomain::Probability,
        })
        .expect("probability");
    let left = optional(predicate, half, &mut arena);
    let right = optional(opposite, half, &mut arena);
    let composed = left
        .probability_then(&right, &mut arena)
        .expect("union bound");
    let result = project(&mut proofs, composed, &mut arena);
    assert!(!result.exact);
    assert!(arena.is_literal(result.lower, "0"));
    assert!(result.upper.is_some());
    assert!(result.unknown.is_empty());
}

#[test]
fn conditional_dependency_metrics_keep_partial_costs_guarded() {
    use super::super::Counts;
    let (mut arena, shot, mut proofs) = context();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::NotEqual, [shot.0, zero])
        .expect("condition");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let division = arena
        .contracted(
            "qamomile.floor_divide",
            ResourceDomain::Integer,
            [one, shot.0],
        )
        .expect("partial span");
    let mut left = Counts::zero(&mut arena).expect("left counts");
    let right = Counts::zero(&mut arena).expect("right counts");
    left.span = SymbolicCount::exact_value(division);
    left.depths
        .as_mut()
        .expect("depths")
        .insert("gates".into(), left.span.clone());
    let conditioned = Counts::conditional(&left, &right, predicate, &BTreeMap::new(), &mut arena)
        .expect("conditioned metrics");
    for count in [
        conditioned.span,
        conditioned.depths.expect("depths")["gates"].clone(),
    ] {
        assert!(matches!(
            arena.node(count.lower),
            Some(Node::Piecewise { .. })
        ));
        let projected = project(&mut proofs, count, &mut arena);
        assert!(!projected.exact && projected.upper.is_none());
        assert_eq!(projected.unknown.as_ref(), &[TOTALITY_REASON]);
        assert!(matches!(
            arena.node(projected.lower),
            Some(Node::Piecewise { .. })
        ));
    }
}

#[test]
fn coinciding_projected_bounds_do_not_upgrade_an_enclosure() {
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("condition");
    let opposite = arena
        .predicate_connective(Predicate::Not, [predicate])
        .expect("negation");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let opposite = recognize(&mut proofs, shot, opposite, &mut arena);
    let mut left = optional(predicate, one, &mut arena);
    left.exact = false;
    let right = optional(opposite, one, &mut arena);
    let sum = left.then(&right, &mut arena).expect("sum");
    let projected = project(&mut proofs, sum, &mut arena);
    assert!(arena.is_literal(projected.lower, "1"));
    assert_eq!(projected.upper, Some(projected.lower));
    assert!(!projected.exact);
}

#[test]
fn custom_nonadditive_metrics_keep_their_existing_branch_enclosures() {
    use super::super::Counts;
    use std::collections::BTreeSet;
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("condition");
    let opposite = arena
        .predicate_connective(Predicate::Not, [predicate])
        .expect("negation");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let opposite = recognize(&mut proofs, shot, opposite, &mut arena);
    let mut present = Counts::zero(&mut arena).expect("present");
    present.work = SymbolicCount::exact_value(one);
    for name in ["custom.span", "custom.peak", "custom.error"] {
        let mut count = SymbolicCount::exact_value(one);
        count.rule_identities = Box::new([name.to_owned()]);
        present.extra_metrics.insert(name.to_owned(), count);
    }
    let absent = Counts::zero(&mut arena).expect("absent");
    let conservative = custom_metric_domains(ResourceDomain::Natural);
    let error_bounds = BTreeSet::from(["custom.error".to_owned()]);
    let mut left = Counts::conditional(&present, &absent, predicate, &conservative, &mut arena)
        .expect("first branch");
    let right = Counts::conditional(&present, &absent, opposite, &conservative, &mut arena)
        .expect("complementary branch");
    left.then(&right, &error_bounds, &mut arena)
        .expect("metric composition");
    let work = project(&mut proofs, left.work, &mut arena);
    assert!(work.exact && arena.is_literal(work.lower, "1"));
    for name in ["custom.span", "custom.peak", "custom.error"] {
        let count = project(&mut proofs, left.extra_metrics[name].clone(), &mut arena);
        assert!(!count.exact);
        assert!(count.upper.is_some() && count.unknown.is_empty());
        assert_eq!(count.rule_identities.as_ref(), &[name]);
        if name == "custom.error" {
            assert!(arena.is_literal(count.upper.expect("error upper bound"), "1"));
        }
    }
}

fn custom_metric_domains(domain: ResourceDomain) -> BTreeMap<String, ResourceDomain> {
    BTreeMap::from([
        ("custom.peak".to_owned(), domain),
        ("custom.span".to_owned(), domain),
    ])
}

fn published_custom_metrics(
    count: SymbolicCount,
    arena: &mut ExpressionArena,
) -> SymbolicLogicalCounts {
    use super::super::SymbolicAnalysisFact;
    use std::collections::BTreeMap;
    let custom = BTreeMap::from([
        ("custom.peak".to_owned(), count.clone()),
        ("custom.span".to_owned(), count.clone()),
    ]);
    let zero = SymbolicCount::exact_value(arena.literal_u128(0).expect("zero"));
    let fact = SymbolicAnalysisFact {
        owner: "sample".into(),
        source: "call".into(),
        opcode: "qcore.call".into(),
        work: zero.clone(),
        opcodes: BTreeMap::new(),
        extra_metrics: custom.clone(),
        families: None,
        measurements: None,
        resets: None,
        cleanup_work: zero.clone(),
    };
    SymbolicLogicalCounts {
        analysis_operation_visits: 0,
        expressions: Box::new([]),
        facts: Box::new([fact]),
        work: count,
        dependency_span: zero.clone(),
        opcodes: BTreeMap::new(),
        extra_metrics: custom,
        families: None,
        depths: None,
        measurements: None,
        resets: None,
        input_qubits: None,
        allocated_qubits: None,
        circuit_qubits: None,
        peak_qubits: None,
        cleanup_work: zero,
    }
}

fn assert_custom_composition_is_unknown(
    result: &SymbolicLogicalCounts,
    original: &SymbolicCount,
    arena: &ExpressionArena,
) {
    for metrics in [&result.extra_metrics, &result.facts[0].extra_metrics] {
        for count in metrics.values() {
            assert!(!count.exact);
            assert!(arena.is_literal(count.lower, "0"));
            assert!(count.upper.is_none());
            assert!(
                count
                    .unknown
                    .iter()
                    .any(|reason| reason == COMPOSITION_REASON)
            );
            assert_eq!(count.rule_identities, original.rule_identities);
            for reason in &original.unknown {
                assert!(count.unknown.contains(reason));
            }
        }
    }
}

#[test]
fn custom_metric_rule_guards_do_not_publish_additive_peak_equalities() {
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let two = arena.literal_u128(2).expect("two");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("condition");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let left = arena
        .piecewise(&[predicate], &[two, one], ResourceDomain::Natural)
        .expect("first rule");
    let right = arena
        .piecewise(&[predicate], &[one, two], ResourceDomain::Natural)
        .expect("second rule");
    let mut combined =
        SymbolicCount::exact_value(arena.add([left, right]).expect("existing composition"));
    combined.rule_identities = Box::new(["rule.left".into(), "rule.right".into()]);
    let mut result = published_custom_metrics(combined.clone(), &mut arena);
    let conservative = custom_metric_domains(ResourceDomain::Natural);
    proofs
        .project(&mut result, &conservative, &mut arena)
        .expect("final projection");
    assert!(result.work.exact && arena.is_literal(result.work.lower, "3"));
    assert_custom_composition_is_unknown(&result, &combined, &arena);
}

#[test]
fn custom_metric_direct_shot_costs_are_unknown_without_predicate_atoms() {
    let (mut arena, shot, mut proofs) = context();
    let mut original = SymbolicCount::exact_value(
        arena
            .add([shot.0, shot.0])
            .expect("existing additive composition"),
    );
    original.rule_identities = Box::new(["rule.first".into(), "rule.second".into()]);
    let mut result = published_custom_metrics(original.clone(), &mut arena);
    let conservative = custom_metric_domains(ResourceDomain::Natural);
    assert!(proofs.atoms.is_empty());
    proofs
        .project(&mut result, &conservative, &mut arena)
        .expect("final projection");
    assert_eq!(result.work, original);
    assert_custom_composition_is_unknown(&result, &original, &arena);
}

#[test]
fn custom_metric_composition_policy_uses_consistent_rule_declarations() {
    let metric = |name: &str, domain, composition| MetricContract {
        identifier: name.to_owned(),
        domain,
        composition,
    };
    let peak = metric(
        "custom.peak",
        ResourceDomain::Natural,
        MetricComposition::PeakLiveness,
    );
    let span = metric(
        "custom.span",
        ResourceDomain::Natural,
        MetricComposition::DependencySpan,
    );
    let additive = metric(
        "custom.work",
        ResourceDomain::Natural,
        MetricComposition::AdditiveWork,
    );
    let error = metric(
        "custom.error",
        ResourceDomain::Probability,
        MetricComposition::ErrorBound,
    );
    let ancilla = metric(
        &format!("{PORTABLE_PREFIX}clean_ancilla_qubits"),
        ResourceDomain::Natural,
        MetricComposition::PeakLiveness,
    );
    assert_eq!(
        unproved_metric_compositions([&peak, &span, &additive, &error, &ancilla, &peak])
            .expect("consistent declarations"),
        custom_metric_domains(ResourceDomain::Natural)
    );
    for conflicting in [
        metric(
            "custom.peak",
            ResourceDomain::Integer,
            MetricComposition::PeakLiveness,
        ),
        metric(
            "custom.peak",
            ResourceDomain::Natural,
            MetricComposition::AdditiveWork,
        ),
    ] {
        for declarations in [[&peak, &conflicting], [&conflicting, &peak]] {
            let error = unproved_metric_compositions(declarations)
                .expect_err("rule order cannot weaken the composition fence");
            assert!(matches!(error, SymbolicResourceError::Unsupported(_)));
            assert!(
                error
                    .to_string()
                    .contains("conflicting domain or composition")
            );
        }
    }
}

#[test]
fn custom_metric_unknowns_preserve_declared_nonnegative_domains_and_evidence() {
    for domain in [ResourceDomain::Natural, ResourceDomain::Probability] {
        let (mut arena, _, mut proofs) = context();
        let mut original = SymbolicCount::exact_value(arena.literal_u128(1).expect("one"));
        original.exact = false;
        original.upper = None;
        original.unknown = Box::new(["existing unknown contribution".into()]);
        original.rule_identities = Box::new(["custom.rule".into()]);
        let mut result = published_custom_metrics(original.clone(), &mut arena);
        let domains = custom_metric_domains(domain);
        proofs.budget.work = 0;
        for _ in 0..2 {
            proofs
                .project(&mut result, &domains, &mut arena)
                .expect("composition fence is independent of predicate budget");
        }
        assert_custom_composition_is_unknown(&result, &original, &arena);
        for metrics in [&result.extra_metrics, &result.facts[0].extra_metrics] {
            for count in metrics.values() {
                assert_eq!(
                    arena.node(count.lower).expect("lower endpoint").domain(),
                    domain
                );
                assert_eq!(count.unknown.len(), 2);
            }
        }
    }
}

#[test]
fn signed_custom_metrics_fail_closed_after_algebra_erases_the_shot_dependency() {
    for domain in [
        ResourceDomain::Integer,
        ResourceDomain::Rational,
        ResourceDomain::Real,
    ] {
        for fact_only in [false, true] {
            let (mut arena, shot, mut proofs) = context();
            let ten = arena.literal_u128(10).expect("ten");
            let negative_one = arena.literal_i64(-1).expect("negative one");
            let negative_shot = arena.multiply([negative_one, shot.0]).expect("negation");
            let second = arena.add([ten, negative_shot]).expect("second cost");
            let sum = arena
                .add([shot.0, second])
                .expect("existing additive composition");
            assert!(arena.is_literal(sum, "10"));
            assert!(!arena.depends_on(sum, shot.0));
            let mut result = published_custom_metrics(SymbolicCount::exact_value(sum), &mut arena);
            if fact_only {
                result.extra_metrics.clear();
            }
            let error = proofs
                .project(&mut result, &custom_metric_domains(domain), &mut arena)
                .expect_err("no signed lower endpoint is proved");
            assert!(matches!(error, SymbolicResourceError::Unsupported(_)));
            let message = error.to_string();
            assert!(message.contains("Q4211"));
            assert!(message.contains("custom.peak"));
            assert!(message.contains("declared domain has no proved lower bound"));
        }
    }
}

#[test]
fn custom_metric_fence_uses_the_ambient_context_even_for_unrelated_costs() {
    for ambient_context in [false, true] {
        let mut arena = ExpressionArena::default();
        let mut proofs = ShotPredicates::default();
        if ambient_context {
            arena.current_shot().expect("ambient context");
        } else {
            arena
                .symbol("sample", "shot", ResourceDomain::Natural, "parameter", &[])
                .expect("unrelated source parameter");
        }
        let original = SymbolicCount::exact_value(arena.literal_u128(2).expect("single cost"));
        let mut result = published_custom_metrics(original.clone(), &mut arena);
        proofs
            .project(
                &mut result,
                &custom_metric_domains(ResourceDomain::Natural),
                &mut arena,
            )
            .expect("final projection");
        if ambient_context {
            assert_custom_composition_is_unknown(&result, &original, &arena);
        } else {
            for metrics in [&result.extra_metrics, &result.facts[0].extra_metrics] {
                assert!(metrics.values().all(|count| *count == original));
            }
        }
    }
}

#[test]
fn final_projection_covers_every_published_metric_and_source_fact() {
    use super::super::{SymbolicAnalysisFact, SymbolicLogicalCounts};
    use std::collections::BTreeMap;
    let (mut arena, shot, mut proofs) = context();
    let one = arena.literal_u128(1).expect("one");
    let predicate = arena
        .predicate_comparison(Predicate::Equal, [shot.0, one])
        .expect("comparison");
    let predicate = recognize(&mut proofs, shot, predicate, &mut arena);
    let count = optional(predicate, one, &mut arena);
    let counts = BTreeMap::from([("metric".to_owned(), count.clone())]);
    let fact = SymbolicAnalysisFact {
        owner: "sample".into(),
        source: "source".into(),
        opcode: "qcore.if".into(),
        work: count.clone(),
        opcodes: counts.clone(),
        extra_metrics: counts.clone(),
        families: Some(counts.clone()),
        measurements: Some(count.clone()),
        resets: Some(count.clone()),
        cleanup_work: count.clone(),
    };
    let mut result = SymbolicLogicalCounts {
        analysis_operation_visits: 0,
        expressions: Box::new([]),
        facts: Box::new([fact]),
        work: count.clone(),
        dependency_span: count.clone(),
        opcodes: counts.clone(),
        extra_metrics: counts.clone(),
        families: Some(counts.clone()),
        depths: Some(counts),
        measurements: Some(count.clone()),
        resets: Some(count.clone()),
        input_qubits: Some(count.clone()),
        allocated_qubits: Some(count.clone()),
        circuit_qubits: Some(count.clone()),
        peak_qubits: Some(count.clone()),
        cleanup_work: count,
    };
    proofs
        .project(&mut result, &BTreeMap::new(), &mut arena)
        .expect("all metric projection");
    let fact = &result.facts[0];
    for count in [
        &result.work,
        &result.dependency_span,
        &result.cleanup_work,
        &result.opcodes["metric"],
        &result.extra_metrics["metric"],
        &result.families.as_ref().expect("families")["metric"],
        &result.depths.as_ref().expect("depths")["metric"],
        result.measurements.as_ref().expect("measurements"),
        result.resets.as_ref().expect("resets"),
        result.input_qubits.as_ref().expect("input width"),
        result.allocated_qubits.as_ref().expect("allocated width"),
        result.circuit_qubits.as_ref().expect("circuit width"),
        result.peak_qubits.as_ref().expect("peak width"),
        &fact.work,
        &fact.cleanup_work,
        &fact.opcodes["metric"],
        &fact.extra_metrics["metric"],
        &fact.families.as_ref().expect("fact families")["metric"],
        fact.measurements.as_ref().expect("fact measurements"),
        fact.resets.as_ref().expect("fact resets"),
    ] {
        assert!(!count.exact);
        assert!(matches!(
            arena.node(count.lower),
            Some(Node::Minimum { .. })
        ));
    }
}
