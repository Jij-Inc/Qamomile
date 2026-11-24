use super::*;
use crate::resource_rules::RuleRelation;

#[test]
fn path_dependent_returns_keep_only_proven_zero_resource_components_exact() {
    let mut arena = ExpressionArena::default();
    let mut counts = Counts::zero(&mut arena).expect("zero counts");
    let one = arena.literal_u128(1).expect("custom cost");
    let custom = SymbolicCount::from_rule(
        "test.classical-cost.v1",
        RuleRelation::Equality,
        one,
        &mut arena,
    )
    .expect("custom rule");
    counts.extra_metrics.insert("test.cost".into(), custom);
    let negative = arena.literal_i64(-1).expect("signed custom cost");
    counts.extra_metrics.insert(
        "test.signed-cost".into(),
        SymbolicCount::exact_value(negative),
    );
    counts.unresolved_control_flow("path-dependent qcore.for return", &arena);
    assert!(counts.work.exact);
    assert!(arena.is_literal(counts.work.lower, "0"));
    let cost = &counts.extra_metrics["test.cost"];
    assert!(!cost.exact);
    assert!(cost.upper.is_none());
    assert_eq!(cost.unknown.as_ref(), ["path-dependent qcore.for return"]);
    assert_eq!(cost.rule_identities.as_ref(), ["test.classical-cost.v1"]);
    let signed = &counts.extra_metrics["test.signed-cost"];
    assert_eq!(signed.lower, negative);
    assert!(signed.upper.is_none());
    assert!(!signed.exact);
}

fn with_test_analyzer(run: impl FnOnce(&mut SymbolicAnalyzer<'_>)) {
    let package = wire::QCorePackageProto::default();
    let mut types = TypeArena::default();
    let operations =
        crate::operations::validate_program_operations_with_types(&package, &mut types)
            .expect("empty operation roots");
    let graph = crate::call_graph::build(&package).expect("empty call graph");
    let mut analyzer = SymbolicAnalyzer {
        functions: HashMap::new(),
        arena: ExpressionArena::default(),
        current_shot: None,
        shot_predicates: shot_predicates::ShotPredicates::default(),
        rules: HashMap::new(),
        error_bound_metrics: BTreeSet::new(),
        unproved_composition_metrics: BTreeMap::new(),
        facts: Vec::new(),
        fact_keys: HashSet::new(),
        timing_aliases: HashMap::new(),
        types: &types,
        operations: &operations,
        verified_calls: &crate::abi::VerifiedCallDimensions::default(),
        analysis_cache: HashMap::new(),
        recurrences: recurrences::RecurrenceAnalysis::new(&graph, &package),
        while_contexts: 0,
        while_functions: HashSet::new(),
        operation_visits: 0,
        observation: None,
    };
    run(&mut analyzer);
}

fn test_block(kind: &str, operands: Vec<u64>) -> wire::BlockProto {
    wire::BlockProto {
        terminator: Some(wire::TerminatorProto {
            kind: kind.into(),
            operands,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn test_value(id: u64) -> wire::ValueProto {
    wire::ValueProto {
        id,
        ..Default::default()
    }
}

fn for_return_operation() -> wire::OperationProto {
    let conditional = wire::OperationProto {
        opcode: "qcore.if".into(),
        operands: vec![4, 4],
        regions: vec![
            wire::RegionProto {
                blocks: vec![wire::BlockProto {
                    arguments: vec![test_value(9)],
                    ..test_block("return", vec![9])
                }],
            },
            wire::RegionProto {
                blocks: vec![wire::BlockProto {
                    arguments: vec![test_value(10)],
                    ..test_block("yield", Vec::new())
                }],
            },
        ],
        ..Default::default()
    };
    wire::OperationProto {
        opcode: "qcore.for".into(),
        operands: vec![0, 1, 2, 6],
        results: vec![test_value(7)],
        regions: vec![
            wire::RegionProto {
                blocks: vec![wire::BlockProto {
                    arguments: vec![test_value(3), test_value(4)],
                    operations: vec![conditional],
                    ..test_block("yield", vec![4])
                }],
            },
            wire::RegionProto {
                blocks: vec![wire::BlockProto {
                    arguments: vec![test_value(8)],
                    ..test_block("yield", vec![8])
                }],
            },
        ],
        ..Default::default()
    }
}

fn for_environment(
    analyzer: &mut SymbolicAnalyzer<'_>,
    exact_upper: Option<i64>,
) -> HashMap<u64, Option<ValueFact>> {
    let mut environment = HashMap::new();
    for (id, value) in [(0, 0), (1, 2), (2, 1)] {
        let value = if id == 1 {
            exact_upper.unwrap_or(value)
        } else {
            value
        };
        let expression = if id == 1 && exact_upper.is_none() {
            analyzer
                .arena
                .symbol("test", "n", ResourceDomain::Integer, "const", &[])
                .expect("symbolic bound")
        } else {
            analyzer.arena.literal_i64(value).expect("bound")
        };
        environment.insert(
            id,
            Some(ValueFact::Scalar(ScalarFact::Int {
                expression,
                exact: if id == 1 { exact_upper } else { Some(value) },
            })),
        );
    }
    environment.insert(6, Some(ValueFact::Scalar(ScalarFact::Bool(None))));
    environment
}

#[test]
fn for_else_exits_keep_the_enclosing_loops_carry_shape() {
    with_test_analyzer(|analyzer| {
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let context = SymbolicFunctionContext {
            owner: "test.loop_else",
            dimensions: &dimensions,
            mode: crate::resource_rules::TransformMode::Direct,
            controls: Some(0),
        };
        for (kind, expected) in [
            ("break", BlockExit::BREAK),
            ("continue", BlockExit::CONTINUE),
        ] {
            for exact_upper in [Some(0), Some(2), None] {
                let mut operation = for_return_operation();
                operation.regions[0].blocks[0].operations.clear();
                operation.regions[1].blocks[0].terminator = test_block(kind, vec![8, 8]).terminator;
                let environment = for_environment(analyzer, exact_upper);
                let summary = analyzer
                    .analyze_for(&operation, &environment, context)
                    .expect("else exits belong to the enclosing loop");
                assert_eq!(summary.exit, expected);
                assert_eq!(summary.values.len(), 2);
                assert!(summary.counts.work.exact);
                assert!(analyzer.arena.is_literal(summary.counts.work.lower, "0"));
            }
        }
    });
}

#[test]
fn a_definite_interpreted_for_else_return_preserves_timing() {
    with_test_analyzer(|analyzer| {
        let mut operation = for_return_operation();
        operation.regions[0].blocks[0].operations.clear();
        operation.regions[1].blocks[0].terminator = test_block("return", vec![8]).terminator;
        let environment = for_environment(analyzer, Some(2));
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let summary = analyzer
            .analyze_for(
                &operation,
                &environment,
                SymbolicFunctionContext {
                    owner: "test.loop_else",
                    dimensions: &dimensions,
                    mode: crate::resource_rules::TransformMode::Direct,
                    controls: Some(0),
                },
            )
            .expect("an unconditional normal-exit return remains exact");
        assert_eq!(summary.exit, BlockExit::RETURN);
        assert_eq!(summary.values, vec![environment[&6].clone()]);
        assert_eq!(summary.timing, Some(Vec::new()));
        assert!(summary.counts.work.exact);
    });
}

#[test]
fn conditional_for_else_exits_retain_inner_fallthrough_values() {
    with_test_analyzer(|analyzer| {
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let context = SymbolicFunctionContext {
            owner: "test.loop_else",
            dimensions: &dimensions,
            mode: crate::resource_rules::TransformMode::Direct,
            controls: Some(0),
        };
        for (kind, escaped) in [
            ("break", BlockExit::BREAK),
            ("continue", BlockExit::CONTINUE),
        ] {
            for exact_upper in [Some(0), Some(2), None] {
                let mut operation = for_return_operation();
                let mut conditional = operation.regions[0].blocks[0].operations.remove(0);
                conditional.operands = vec![8, 8];
                conditional.regions[0].blocks[0].terminator =
                    test_block(kind, vec![9, 9]).terminator;
                operation.regions[1].blocks[0].operations.push(conditional);
                let environment = for_environment(analyzer, exact_upper);
                let summary = analyzer
                    .analyze_for(&operation, &environment, context)
                    .expect("conditional else exits preserve normal completion");
                assert_eq!(summary.exit, BlockExit::YIELD.union(escaped));
                assert_eq!(summary.values, vec![None]);
                assert!(summary.counts.work.exact);
                assert!(analyzer.arena.is_literal(summary.counts.work.lower, "0"));
                assert!(summary.timing.is_none());
            }
        }
    });
}

#[test]
fn inner_body_breaks_and_else_exits_keep_distinct_targets() {
    with_test_analyzer(|analyzer| {
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let context = SymbolicFunctionContext {
            owner: "test.loop_else",
            dimensions: &dimensions,
            mode: crate::resource_rules::TransformMode::Direct,
            controls: Some(0),
        };
        for (kind, escaped) in [
            ("break", BlockExit::BREAK),
            ("continue", BlockExit::CONTINUE),
        ] {
            for exact_upper in [Some(2), None] {
                let mut operation = for_return_operation();
                operation.regions[0].blocks[0].operations[0].regions[0].blocks[0].terminator =
                    test_block("break", vec![9]).terminator;
                operation.regions[1].blocks[0].terminator = test_block(kind, vec![8, 8]).terminator;
                let environment = for_environment(analyzer, exact_upper);
                let summary = analyzer
                    .analyze_for(&operation, &environment, context)
                    .expect("body breaks bypass the outward else exit");
                assert_eq!(summary.exit, BlockExit::YIELD.union(escaped));
                assert_eq!(summary.values, vec![None]);
                assert!(summary.counts.work.exact);
            }
        }
    });
}

#[test]
fn for_body_returns_preserve_fallthrough_and_zero_costs() {
    with_test_analyzer(|analyzer| {
        let operation = for_return_operation();
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let context = SymbolicFunctionContext {
            owner: "test.loop_return",
            dimensions: &dimensions,
            mode: crate::resource_rules::TransformMode::Direct,
            controls: Some(0),
        };
        for exact_upper in [Some(2), None] {
            let environment = for_environment(analyzer, exact_upper);
            let summary = analyzer
                .analyze_for(&operation, &environment, context)
                .expect("path-dependent return remains analyzable");
            assert_eq!(summary.exit, BlockExit::YIELD.union(BlockExit::RETURN));
            assert_eq!(summary.values, vec![None]);
            assert!(summary.counts.work.exact);
            assert!(analyzer.arena.is_literal(summary.counts.work.lower, "0"));
            assert!(summary.counts.work.unknown.is_empty());
            assert!(summary.timing.is_none());

            let mut returning_else = operation.clone();
            returning_else.regions[1].blocks[0].terminator =
                test_block("return", vec![8]).terminator;
            let summary = analyzer
                .analyze_for(&returning_else, &environment, context)
                .expect("both normal completion and early exits return");
            assert_eq!(summary.exit, BlockExit::RETURN);
            assert_eq!(summary.values, vec![None]);
            assert!(summary.counts.work.exact);
        }
    });
}

fn test_constant(id: u64, value: wire::literal_proto::Value) -> wire::OperationProto {
    wire::OperationProto {
        opcode: "qcore.constant".into(),
        results: vec![test_value(id)],
        attributes: vec![wire::AttributeProto {
            name: "value".into(),
            value: Some(wire::LiteralProto { value: Some(value) }),
        }],
        ..Default::default()
    }
}

#[test]
fn a_later_iteration_return_does_not_erase_an_earlier_return_value() {
    with_test_analyzer(|analyzer| {
        let mut operation = for_return_operation();
        let body = &mut operation.regions[0].blocks[0];
        let early_branch = &mut body.operations[0].regions[0].blocks[0];
        early_branch.operations = vec![test_constant(
            23,
            wire::literal_proto::Value::BoolValue(true),
        )];
        early_branch.terminator = test_block("return", vec![23]).terminator;
        let compare = wire::OperationProto {
            opcode: "classical.compare_chain".into(),
            operands: vec![3, 20],
            results: vec![test_value(21)],
            attributes: vec![wire::AttributeProto {
                name: "operators".into(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::TupleValue(
                        wire::literal_proto::Sequence {
                            values: vec![wire::LiteralProto {
                                value: Some(wire::literal_proto::Value::StringValue("eq".into())),
                            }],
                        },
                    )),
                }),
            }],
            ..Default::default()
        };
        let final_return = wire::OperationProto {
            opcode: "qcore.if".into(),
            operands: vec![21],
            regions: vec![
                wire::RegionProto {
                    blocks: vec![wire::BlockProto {
                        operations: vec![test_constant(
                            22,
                            wire::literal_proto::Value::BoolValue(false),
                        )],
                        ..test_block("return", vec![22])
                    }],
                },
                wire::RegionProto {
                    blocks: vec![test_block("yield", Vec::new())],
                },
            ],
            ..Default::default()
        };
        body.operations.splice(
            0..0,
            [
                test_constant(20, wire::literal_proto::Value::IntValue(1)),
                compare,
                final_return,
            ],
        );
        let environment = for_environment(analyzer, Some(2));
        let dimensions = SymbolicDimensionBindings {
            values: HashMap::new(),
        };
        let summary = analyzer
            .analyze_for(
                &operation,
                &environment,
                SymbolicFunctionContext {
                    owner: "test.loop_return",
                    dimensions: &dimensions,
                    mode: crate::resource_rules::TransformMode::Direct,
                    controls: Some(0),
                },
            )
            .expect("both return values remain possible");
        assert_eq!(summary.exit, BlockExit::RETURN);
        assert_eq!(summary.values, vec![None]);
    });
}

#[test]
fn small_permutation_resource_bound_is_bounded_before_expansion() {
    assert_eq!(small_permutation_step_bound(1).expect("one bit"), 1);
    assert_eq!(small_permutation_step_bound(2).expect("two bits"), 27);
    assert_eq!(small_permutation_step_bound(3).expect("three bits"), 175);
    for width in [0, 4, 64, u64::MAX] {
        assert!(small_permutation_step_bound(width).is_err());
    }
}

#[test]
fn parametric_dependency_scheduler_joins_forks_before_their_successor() {
    let mut arena = ExpressionArena::default();
    let left = arena
        .symbol("example", "left", ResourceDomain::Natural, "const", &[])
        .expect("left duration");
    let right = arena
        .symbol("example", "right", ResourceDomain::Natural, "const", &[])
        .expect("right duration");
    let two = arena.literal_u128(2).expect("two");
    let mut scheduler =
        ParametricDependencyScheduler::new(&mut arena).expect("dependency scheduler");

    scheduler.step(&[0], left, &mut arena).expect("left branch");
    scheduler
        .step(&[1], right, &mut arena)
        .expect("right branch");
    scheduler
        .step(&[0, 1], two, &mut arena)
        .expect("joined successor");

    assert!(matches!(
        arena.node(scheduler.makespan()),
        Some(ResourceExpressionNode::Add { arguments, .. })
            if arguments.iter().any(|argument| arena.is_literal(*argument, "2"))
                && arguments.iter().any(|argument| matches!(
                    arena.node(*argument),
                    Some(ResourceExpressionNode::Maximum { .. })
                ))
    ));
}

#[test]
fn parametric_dependency_scheduler_rejects_non_natural_durations() {
    let mut arena = ExpressionArena::default();
    let duration = arena
        .symbol("example", "time", ResourceDomain::Integer, "const", &[])
        .expect("integer duration");
    let mut scheduler =
        ParametricDependencyScheduler::new(&mut arena).expect("dependency scheduler");

    assert!(matches!(
        scheduler.step(&[0], duration, &mut arena),
        Err(SymbolicResourceError::Unsupported(reason))
            if reason == "parametric schedule duration must be natural"
    ));
}

#[test]
fn symbolic_polynomial_normalization_collects_like_terms() {
    let mut arena = ExpressionArena::default();
    let x = arena
        .symbol(
            "example",
            "x",
            ResourceDomain::Natural,
            "const",
            &["x >= 1"],
        )
        .expect("symbol");
    let two = arena.literal_u128(2).expect("two");
    let three = arena.literal_u128(3).expect("three");
    let four = arena.literal_u128(4).expect("four");
    let twice = arena.multiply([two, x]).expect("twice x");
    let thrice = arena.multiply([three, x]).expect("thrice x");
    let normalized = arena.add([twice, thrice, four]).expect("normalized sum");

    let Some(ResourceExpressionNode::Add { arguments, .. }) = arena.node(normalized) else {
        panic!("normalization did not retain a nontrivial sum");
    };
    assert_eq!(arguments.len(), 2);
    assert!(
        arguments
            .iter()
            .any(|argument| arena.is_literal(*argument, "4"))
    );
    assert!(arguments.iter().any(|argument| {
        matches!(
            arena.node(*argument),
            Some(ResourceExpressionNode::Multiply { arguments, .. })
                if arguments.contains(&x)
                    && arguments.iter().any(|factor| arena.is_literal(*factor, "5"))
        )
    }));
}

#[test]
fn range_normalization_requires_a_nonnegative_stop() {
    let mut arena = ExpressionArena::default();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let natural = arena
        .symbol(
            "example",
            "n",
            ResourceDomain::Natural,
            "const",
            &["n >= 1"],
        )
        .expect("natural symbol");
    let integer = arena
        .symbol("example", "k", ResourceDomain::Integer, "const", &[])
        .expect("integer symbol");

    assert_eq!(
        arena
            .contracted(
                "qamomile.range_trip_count",
                ResourceDomain::Natural,
                [zero, natural, one],
            )
            .expect("natural range"),
        natural
    );
    let two = arena.literal_u128(2).expect("two");
    let half = arena
        .contracted(
            "qamomile.floor_divide",
            ResourceDomain::Integer,
            [natural, two],
        )
        .expect("half dimension");
    let half_range = arena
        .contracted(
            "qamomile.range_trip_count",
            ResourceDomain::Natural,
            [zero, half, one],
        )
        .expect("half range");
    assert!(matches!(
        arena.node(half_range),
        Some(ResourceExpressionNode::Contracted {
            name,
            domain: ResourceDomain::Natural,
            ..
        }) if name == "qamomile.floor_divide"
    ));
    let unresolved = arena
        .contracted(
            "qamomile.range_trip_count",
            ResourceDomain::Natural,
            [zero, integer, one],
        )
        .expect("integer range");
    assert!(matches!(
        arena.node(unresolved),
        Some(ResourceExpressionNode::Contracted { name, .. })
            if name == "qamomile.range_trip_count"
    ));
}

#[test]
fn transform_modes_form_a_closed_adjoint_control_algebra() {
    assert_eq!(TransformMode::Direct.adjoint(), TransformMode::Adjoint);
    assert_eq!(TransformMode::Adjoint.adjoint(), TransformMode::Direct);
    assert_eq!(
        TransformMode::Controlled.adjoint(),
        TransformMode::ControlledAdjoint
    );
    assert_eq!(
        TransformMode::ControlledAdjoint.adjoint(),
        TransformMode::Controlled
    );
    assert_eq!(
        TransformMode::Adjoint.controlled(),
        TransformMode::ControlledAdjoint
    );
    assert_eq!(
        TransformMode::ControlledAdjoint.controlled(),
        TransformMode::ControlledAdjoint
    );
    assert_eq!(
        TransformMode::Controlled.controlled(),
        TransformMode::Controlled
    );
}

#[test]
fn logical_counts_own_the_complete_unknown_contributor_union() {
    fn unknown(reason: &str) -> SymbolicCount {
        SymbolicCount {
            lower: ResourceExpressionId(0),
            upper: None,
            exact: false,
            unknown: Box::new([reason.to_owned(), "shared".to_owned()]),
            rule_identities: Box::new([]),
        }
    }

    let counts = SymbolicLogicalCounts {
        analysis_operation_visits: 0,
        expressions: Box::new([]),
        facts: Box::new([]),
        work: unknown("work"),
        dependency_span: unknown("span"),
        opcodes: BTreeMap::from([("quantum.h".to_owned(), unknown("opcode"))]),
        extra_metrics: BTreeMap::from([("example.metric".to_owned(), unknown("metric"))]),
        families: Some(BTreeMap::from([("gates".to_owned(), unknown("family"))])),
        measurements: Some(unknown("measurements")),
        resets: Some(unknown("resets")),
        input_qubits: Some(unknown("input")),
        allocated_qubits: Some(unknown("allocated")),
        circuit_qubits: Some(unknown("circuit")),
        peak_qubits: Some(unknown("peak")),
        depths: Some(BTreeMap::from([("depth".to_owned(), unknown("depth"))])),
        cleanup_work: unknown("cleanup"),
    };

    assert_eq!(
        counts.unknown_contributors(),
        [
            "allocated",
            "circuit",
            "cleanup",
            "depth",
            "family",
            "input",
            "measurements",
            "metric",
            "opcode",
            "peak",
            "resets",
            "shared",
            "span",
            "work",
        ]
    );
}

#[test]
fn scalar_span_composes_without_an_exact_lane_schedule() {
    let mut arena = ExpressionArena::default();
    let count = arena.literal_u128(1).expect("one");
    let one = Counts::quantum("quantum.h", "quantum.h", None, count, false, &mut arena)
        .expect("one step");
    let mut sequence = one.clone();
    sequence
        .then(&one, &BTreeSet::new(), &mut arena)
        .expect("sequence");

    assert_eq!(evaluate(sequence.span.lower, &arena), 1);
    assert_eq!(evaluate(sequence.span.upper.expect("upper"), &arena), 2);
    assert!(!sequence.span.exact);
    let sequence_gate_depth = &sequence.depths.as_ref().expect("depths")["gates"];
    assert_eq!(evaluate(sequence_gate_depth.lower, &arena), 1);
    assert_eq!(
        evaluate(sequence_gate_depth.upper.expect("upper"), &arena),
        2
    );
    assert!(!sequence_gate_depth.exact);

    let alternatives =
        Counts::branch(&[one.clone(), sequence.clone()], &mut arena).expect("branch enclosure");
    assert_eq!(evaluate(alternatives.span.lower, &arena), 1);
    assert_eq!(evaluate(alternatives.span.upper.expect("upper"), &arena), 2);
    assert!(!alternatives.span.exact);

    let trip_count = arena.literal_u128(3).expect("trip count");
    let binder = arena
        .symbol(
            "test:loop",
            "index",
            ResourceDomain::Natural,
            "loop_iteration",
            &[],
        )
        .expect("binder");
    let repeated = one
        .repeat(trip_count, binder, &BTreeSet::new(), &mut arena)
        .expect("repeated span");
    assert_eq!(evaluate(repeated.span.lower, &arena), 1);
    assert_eq!(evaluate(repeated.span.upper.expect("upper"), &arena), 3);
    assert!(!repeated.span.exact);
    let repeated_gate_depth = &repeated.depths.as_ref().expect("depths")["gates"];
    assert_eq!(evaluate(repeated_gate_depth.lower, &arena), 1);
    assert_eq!(
        evaluate(repeated_gate_depth.upper.expect("upper"), &arena),
        3
    );
    assert!(!repeated_gate_depth.exact);

    let five = arena.literal_u128(5).expect("rule expression");
    let upper = SymbolicCount::from_rule("upper", RuleRelation::UpperBound, five, &mut arena)
        .expect("upper span rule")
        .dependency_then(&one.span, &mut arena)
        .expect("upper sequence");
    assert_eq!(evaluate(upper.lower, &arena), 1);
    assert_eq!(evaluate(upper.upper.expect("upper"), &arena), 6);
    assert!(!upper.exact);

    let lower = SymbolicCount::from_rule("lower", RuleRelation::LowerBound, five, &mut arena)
        .expect("lower span rule")
        .dependency_then(&one.span, &mut arena)
        .expect("lower sequence");
    assert_eq!(evaluate(lower.lower, &arena), 5);
    assert_eq!(lower.upper, None);
    assert!(!lower.exact);
}

#[test]
fn error_bound_composition_is_selected_by_metric_contract() {
    let mut arena = ExpressionArena::default();
    let one = SymbolicCount::exact_value(arena.literal_u128(1).expect("one"));
    let contribution = SymbolicCount::bounded_by(&one, &mut arena).expect("error bound");
    let identifier = "example.output_error".to_owned();
    let mut left = Counts::zero(&mut arena).expect("left counts");
    left.extra_metrics
        .insert(identifier.clone(), contribution.clone());
    let mut right = Counts::zero(&mut arena).expect("right counts");
    right.extra_metrics.insert(identifier.clone(), contribution);

    left.then(&right, &BTreeSet::from([identifier.clone()]), &mut arena)
        .expect("compose error bounds");

    let composed = &left.extra_metrics[&identifier];
    assert_eq!(evaluate(composed.lower, &arena), 0);
    assert_eq!(evaluate(composed.upper.expect("upper"), &arena), 1);
}

#[test]
fn exact_zero_error_composition_keeps_one_canonical_endpoint() {
    let mut arena = ExpressionArena::default();
    let zero = SymbolicCount::exact_value(arena.literal_u128(0).expect("zero"));
    let sequential = zero
        .probability_then(&zero, &mut arena)
        .expect("sequential exact error");
    assert!(sequential.exact);
    assert_eq!(sequential.upper, Some(sequential.lower));

    let trip_count = arena.literal_u128(3).expect("trip count");
    let binder = arena
        .symbol(
            "test:error-loop",
            "index",
            ResourceDomain::Natural,
            "loop_iteration",
            &[],
        )
        .expect("binder");
    let repeated = zero
        .probability_repeat(trip_count, binder, &mut arena)
        .expect("repeated exact error");
    assert!(repeated.exact);
    assert_eq!(repeated.upper, Some(repeated.lower));
}

#[test]
fn symbolic_counts_compose_canonical_rule_usage() {
    let mut arena = ExpressionArena::default();
    let one = arena.literal_u128(1).expect("one");
    let two = arena.literal_u128(2).expect("two");
    let left =
        SymbolicCount::from_rule("example.z-rule.v1", RuleRelation::Equality, one, &mut arena)
            .expect("left rule");
    let right = SymbolicCount::from_rule(
        "example.a-rule.v1",
        RuleRelation::UpperBound,
        two,
        &mut arena,
    )
    .expect("right rule");
    let same_value =
        SymbolicCount::from_rule("example.m-rule.v1", RuleRelation::Equality, one, &mut arena)
            .expect("same-value rule");

    let sequence = left.clone().then(&right, &mut arena).expect("sequence");
    let branch = SymbolicCount::branch(&[right, left.clone()], &mut arena).expect("branch");
    let equal_branch =
        SymbolicCount::branch(&[left.clone(), same_value], &mut arena).expect("equal branch");
    let zero = arena.literal_u128(0).expect("zero");
    let binder = arena
        .symbol(
            "test:rule-usage",
            "index",
            ResourceDomain::Natural,
            "loop_iteration",
            &[],
        )
        .expect("binder");
    let skipped = left.repeat(zero, binder, &mut arena).expect("skipped rule");

    assert_eq!(
        sequence.rule_identities(),
        ["example.a-rule.v1", "example.z-rule.v1"]
    );
    assert_eq!(branch.rule_identities(), sequence.rule_identities());
    assert_eq!(
        equal_branch.rule_identities(),
        ["example.m-rule.v1", "example.z-rule.v1"]
    );
    assert!(skipped.rule_identities().is_empty());
}

#[test]
fn symbolic_count_relations_are_classified_natively() {
    let mut arena = ExpressionArena::default();
    let zero = arena.literal_u128(0).expect("zero");
    let one = arena.literal_u128(1).expect("one");
    let two = arena.literal_u128(2).expect("two");
    let exact = SymbolicCount::exact_value(one);
    let lower = SymbolicCount::from_rule(
        "example.lower.v1",
        RuleRelation::LowerBound,
        one,
        &mut arena,
    )
    .expect("lower bound");
    let upper = SymbolicCount::from_rule(
        "example.upper.v1",
        RuleRelation::UpperBound,
        two,
        &mut arena,
    )
    .expect("upper bound");
    let enclosure = SymbolicCount {
        lower: one,
        upper: Some(two),
        exact: false,
        unknown: Box::new([]),
        rule_identities: Box::new([]),
    };
    let partial = SymbolicCount::unresolved("opaque contribution".to_owned(), &mut arena)
        .expect("partial enclosure");
    let expressions = arena.projected_nodes();

    assert_eq!(exact.relation(), SymbolicCountRelation::Equality);
    assert_eq!(lower.relation(), SymbolicCountRelation::LowerBound);
    assert_eq!(upper.relation(), SymbolicCountRelation::Enclosure);
    assert_eq!(enclosure.relation(), SymbolicCountRelation::Enclosure);
    assert_eq!(partial.relation(), SymbolicCountRelation::PartialEnclosure);
    assert_eq!(
        upper.dependency_span_relation(&expressions),
        SymbolicCountRelation::UpperBound
    );
    assert_eq!(
        enclosure.dependency_span_relation(&expressions),
        SymbolicCountRelation::Enclosure
    );
    assert_eq!(
        SymbolicCount {
            lower: zero,
            upper: Some(two),
            exact: false,
            unknown: Box::new(["unknown".to_owned()]),
            rule_identities: Box::new([]),
        }
        .dependency_span_relation(&expressions),
        SymbolicCountRelation::PartialEnclosure
    );
}

#[test]
fn projected_expression_metadata_stays_closed_and_typed() {
    let mut arena = ExpressionArena::default();
    let integer = arena.literal_i64(-3).expect("integer");
    let symbol = arena
        .symbol(
            "test:typed-projection",
            "n",
            ResourceDomain::Natural,
            "parameter",
            &["nonnegative"],
        )
        .expect("symbol");
    let predicate = arena
        .predicate_comparison(ResourcePredicateKind::LessEqual, [integer, symbol])
        .expect("predicate");
    let rational = arena
        .intern(ResourceExpressionNode::RationalLiteral {
            numerator: "1".to_owned(),
            denominator: "3".to_owned(),
            domain: ResourceDomain::Rational,
        })
        .expect("rational");
    let nodes = arena.projected_nodes();

    let integer_node = &nodes[integer.index().expect("integer index")];
    assert_eq!(integer_node.kind(), ResourceExpressionKind::Literal);
    assert_eq!(integer_node.domain(), ResourceDomain::Integer);
    assert_eq!(
        integer_node.literal_kind(),
        Some(ResourceLiteralKind::Integer)
    );

    let symbol_metadata = nodes[symbol.index().expect("symbol index")]
        .symbol()
        .expect("symbol metadata");
    assert_eq!(symbol_metadata.scope(), "test:typed-projection");
    assert_eq!(symbol_metadata.name(), "n");
    assert_eq!(symbol_metadata.domain(), ResourceDomain::Natural);
    assert_eq!(symbol_metadata.origin(), "parameter");
    assert_eq!(symbol_metadata.constraints(), ["nonnegative"]);

    let predicate_node = &nodes[predicate.index().expect("predicate index")];
    assert_eq!(
        predicate_node.kind(),
        ResourceExpressionKind::Predicate(ResourcePredicateKind::LessEqual)
    );
    assert_eq!(predicate_node.domain(), ResourceDomain::Boolean);
    assert_eq!(predicate_node.arguments(), [integer, symbol]);

    let rational_metadata = nodes[rational.index().expect("rational index")]
        .rational_literal()
        .expect("rational metadata");
    assert_eq!(rational_metadata.numerator(), "1");
    assert_eq!(rational_metadata.denominator(), "3");
}

#[test]
fn symbolic_power_accounts_for_negative_integer_exponents() {
    let mut arena = ExpressionArena::default();
    let base = arena.literal_u128(2).expect("base");
    let exponent = arena
        .symbol("test", "n", ResourceDomain::Integer, "const", &[])
        .expect("exponent");

    let result = arena.power(base, exponent).expect("power");

    assert_eq!(
        arena.node(result).expect("result").domain(),
        ResourceDomain::Rational
    );

    let negative = arena.literal_i64(-3).expect("negative exponent");
    let reciprocal = arena.power(base, negative).expect("reciprocal power");
    assert_eq!(
        arena.node(reciprocal).expect("reciprocal").domain(),
        ResourceDomain::Rational
    );
}

#[test]
fn schedules_a_symbolic_full_width_lane_map_without_unrolling() {
    let mut arena = ExpressionArena::default();
    let width = arena.literal_u128(100_000).expect("width");
    let binder = arena
        .symbol(
            "test:loop",
            "index",
            ResourceDomain::Natural,
            "loop_iteration",
            &[],
        )
        .expect("binder");
    let root = SymbolicTimingRoot {
        scope: "test.main".to_owned(),
        root: 0,
        projections: Vec::new(),
        instance: Vec::new(),
    };
    let full = || SymbolicTimingFootprint {
        root: root.clone(),
        selectors: None,
        width: Some(width),
    };
    let lane = || SymbolicTimingFootprint {
        root: root.clone(),
        selectors: Some(vec![binder].into_boxed_slice()),
        width: Some(width),
    };
    let empty = symbolic_unit_durations(&[], &mut arena).expect("empty durations");
    let clifford = symbolic_unit_durations(&["clifford_gates", "gates"], &mut arena)
        .expect("clifford durations");
    let t = symbolic_unit_durations(&["gates", "t_gates"], &mut arena).expect("T durations");
    let measurement =
        symbolic_unit_durations(&["measurements"], &mut arena).expect("measurement durations");
    let nodes = vec![
        SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: vec![full()].into_boxed_slice(),
            durations: empty,
        }),
        SymbolicTimingNode::Repeat {
            body: vec![
                SymbolicTimingNode::Step(SymbolicTimingStep {
                    footprints: vec![lane()].into_boxed_slice(),
                    durations: clifford,
                }),
                SymbolicTimingNode::Step(SymbolicTimingStep {
                    footprints: vec![lane()].into_boxed_slice(),
                    durations: t,
                }),
            ]
            .into_boxed_slice(),
            binder,
            trip_count: width,
        },
        SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: vec![full()].into_boxed_slice(),
            durations: measurement,
        }),
    ];

    let depths = schedule_symbolic_depths(&nodes, &mut arena).expect("depths");

    assert_eq!(evaluate(depths["span"].lower, &arena), 4);
    assert_eq!(evaluate(depths["gates"].lower, &arena), 2);
    assert_eq!(evaluate(depths["clifford_gates"].lower, &arena), 1);
    assert_eq!(evaluate(depths["t_gates"].lower, &arena), 1);
    assert_eq!(evaluate(depths["measurements"].lower, &arena), 1);
}

#[test]
fn schedules_one_dynamic_selector_per_independent_owner() {
    let mut arena = ExpressionArena::default();
    let width = arena.literal_u128(8).expect("width");
    let selector = arena
        .symbol("test", "target", ResourceDomain::Natural, "parameter", &[])
        .expect("selector");
    let other_selector = arena
        .symbol(
            "test",
            "other_target",
            ResourceDomain::Natural,
            "parameter",
            &[],
        )
        .expect("other selector");
    let timing_root = |root| SymbolicTimingRoot {
        scope: "test.main".to_owned(),
        root,
        projections: Vec::new(),
        instance: Vec::new(),
    };
    let step = |root, selector, durations| {
        SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: vec![SymbolicTimingFootprint {
                root: timing_root(root),
                selectors: Some(vec![selector].into_boxed_slice()),
                width: Some(width),
            }]
            .into_boxed_slice(),
            durations,
        })
    };
    let first =
        symbolic_unit_durations(&["clifford_gates", "gates"], &mut arena).expect("first durations");
    let second =
        symbolic_unit_durations(&["gates", "t_gates"], &mut arena).expect("second durations");

    let parallel = schedule_symbolic_depths(
        &[
            step(0, selector, first.clone()),
            step(1, other_selector, second.clone()),
        ],
        &mut arena,
    )
    .expect("parallel dynamic selectors");
    assert_eq!(evaluate(parallel["gates"].lower, &arena), 1);

    let serial = schedule_symbolic_depths(
        &[
            step(0, selector, first.clone()),
            step(0, selector, second.clone()),
        ],
        &mut arena,
    )
    .expect("repeated dynamic selector");
    assert_eq!(evaluate(serial["gates"].lower, &arena), 2);

    assert!(matches!(
        schedule_symbolic_depths(
            &[
                step(0, selector, first),
                step(0, other_selector, second),
            ],
            &mut arena,
        ),
        Err(SymbolicResourceError::Unsupported(reason))
            if reason == "dynamic timing selector may alias another lane"
    ));
}

fn evaluate(expression: ExprId, arena: &ExpressionArena) -> u128 {
    match arena.node(expression).expect("expression") {
        ResourceExpressionNode::Literal { value, .. } => value.parse().expect("literal"),
        ResourceExpressionNode::Add { arguments, .. } => {
            arguments.iter().map(|item| evaluate(*item, arena)).sum()
        }
        ResourceExpressionNode::Multiply { arguments, .. } => arguments
            .iter()
            .map(|item| evaluate(*item, arena))
            .product(),
        ResourceExpressionNode::Minimum { arguments, .. } => arguments
            .iter()
            .map(|item| evaluate(*item, arena))
            .min()
            .expect("minimum"),
        ResourceExpressionNode::Maximum { arguments, .. } => arguments
            .iter()
            .map(|item| evaluate(*item, arena))
            .max()
            .expect("maximum"),
        other => panic!("unexpected expression {other:?}"),
    }
}
