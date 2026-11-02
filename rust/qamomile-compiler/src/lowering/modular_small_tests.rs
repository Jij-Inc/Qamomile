//! Bounded complete-domain witnesses for generated small modular arithmetic.

use super::*;

fn apply(plan: &ModularGatePlan, mut state: u64) -> u64 {
    for step in plan.steps() {
        if step
            .controls()
            .iter()
            .any(|control| state & (1 << control) == 0)
        {
            continue;
        }
        let wires = step.wires();
        if step.gate() == ModularGate::Swap {
            if ((state >> wires[0]) ^ (state >> wires[1])) & 1 != 0 {
                state ^= (1 << wires[0]) | (1 << wires[1]);
            }
            continue;
        }
        assert!(matches!(
            step.gate(),
            ModularGate::X | ModularGate::Cx | ModularGate::Ccx
        ));
        if wires[..wires.len() - 1]
            .iter()
            .all(|control| state & (1 << control) != 0)
        {
            state ^= 1 << wires[wires.len() - 1];
        }
    }
    state
}

#[test]
fn small_children_match_the_total_controlled_unitary_and_schoolbook() {
    for width in 1..=3 {
        for modulus in 1..=(1 << width) {
            let domain = CheckedModulus::new(width, modulus).unwrap();
            for factor in 0..modulus {
                let Some(unit) = domain.unit(factor) else {
                    continue;
                };
                for inverse in [false, true] {
                    let effective = if inverse { unit.inverse() } else { unit }.residue();
                    let schoolbook =
                        LoweringConfig::from_labels(&[], &["mod_mul.schoolbook_ripple".to_owned()])
                            .unwrap();
                    let ModularRealizationPlan::Gates(reference) =
                        plan_mod_pow(width, 1, factor, modulus, inverse, 1, schoolbook).unwrap()
                    else {
                        panic!("expected portable plan")
                    };
                    for capabilities in [vec![], vec!["mcx".to_owned()]] {
                        let config = LoweringConfig::from_labels(&capabilities, &[]).unwrap();
                        let ModularRealizationPlan::Gates(plan) =
                            plan_mod_pow(width, 1, factor, modulus, inverse, 1, config).unwrap()
                        else {
                            panic!("expected portable plan")
                        };
                        assert!(plan.steps().len() <= 175 * 3);
                        if config.capabilities.contains(RealizationCapabilities::MCX) {
                            assert_eq!(plan.scratch_width(), 0);
                        }
                        for target in 0..(1_u64 << width) {
                            for exponent in 0..=1 {
                                for control in 0..=1 {
                                    let source =
                                        target | (exponent << width) | (control << (width + 1));
                                    let output = if exponent == 1
                                        && control == 1
                                        && target < modulus.unsigned_abs()
                                    {
                                        target * effective.unsigned_abs() % modulus.unsigned_abs()
                                    } else {
                                        target
                                    };
                                    let expected =
                                        output | (exponent << width) | (control << (width + 1));
                                    assert_eq!(apply(&plan, source), expected);
                                    assert_eq!(apply(&reference, source), expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn retained_schoolbook_swaps_preserve_all_residues_and_clean_scratch() {
    for width in 1..=3 {
        let config = LoweringConfig::from_labels(
            &["mcx".to_owned(), "controlled_swap".to_owned()],
            &["mod_mul.schoolbook_ripple".to_owned()],
        )
        .unwrap();
        for modulus in 1..=(1 << width) {
            let domain = CheckedModulus::new(width, modulus).unwrap();
            for factor in 0..modulus {
                let Some(unit) = domain.unit(factor) else {
                    continue;
                };
                for inverse in [false, true] {
                    let effective = if inverse { unit.inverse() } else { unit }.residue();
                    let ModularRealizationPlan::Gates(plan) =
                        plan_mod_pow(width, 1, factor, modulus, inverse, 1, config).unwrap()
                    else {
                        panic!("expected gate plan")
                    };
                    for source in 0..(1 << (width + 2)) {
                        let target = source & ((1 << width) - 1);
                        let expected_target =
                            if source >> width == 3 && target < modulus.unsigned_abs() {
                                target * effective.unsigned_abs() % modulus.unsigned_abs()
                            } else {
                                target
                            };
                        assert_eq!(
                            apply(&plan, source),
                            (source & !((1 << width) - 1)) | expected_target
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn identity_suffixes_allocate_nothing_and_selection_honors_the_threshold() {
    let native_controls = LoweringConfig::from_labels(&["mcx".to_owned()], &[]).unwrap();
    for exponent_width in [1, 2, 8] {
        let ModularRealizationPlan::Gates(plan) =
            plan_mod_pow(2, exponent_width, 2, 3, false, 0, native_controls).unwrap()
        else {
            panic!("expected gate plan")
        };
        assert_eq!(plan.steps().len(), 9);
        assert_eq!(plan.scratch_width(), 0);
        assert_eq!(
            plan.child_realizations(),
            &[Realization::ArithmeticPortable]
        );
    }
    for width in [1, 2, 3] {
        assert_eq!(
            native_controls.select_mod_mul_child(width).kind(),
            Realization::ArithmeticPortable
        );
    }
    for width in [4, 9, 62, u64::MAX] {
        assert_eq!(
            native_controls.select_mod_mul_child(width).kind(),
            Realization::ModMulSchoolbookRipple
        );
    }
    let schoolbook =
        LoweringConfig::from_labels(&[], &["mod_mul.schoolbook_ripple".to_owned()]).unwrap();
    assert_eq!(
        schoolbook.select_mod_mul_child(2).kind(),
        Realization::ModMulSchoolbookRipple
    );
    assert_eq!(
        LoweringConfig::EMPTY
            .select(RealizationFamily::ModMul)
            .kind(),
        Realization::ArithmeticPortable
    );
    for modulus in [0, -1, 9] {
        assert!(plan_mod_pow(3, 1, 1, modulus, false, 0, native_controls).is_err());
    }
    assert!(plan_mod_pow(3, 0, 1, 3, false, 0, native_controls).is_err());
    assert!(plan_mod_pow(3, u64::MAX, 2, 3, false, 0, native_controls).is_err());
    let unit = plan_schoolbook_mod_mul(2, 0, 1, false, 0, RealizationCapabilities::EMPTY).unwrap();
    assert!(unit.steps().is_empty());
    assert_eq!(unit.scratch_width(), 0);
}
