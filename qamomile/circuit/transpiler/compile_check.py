from qamomile.circuit.ir.block_value import BlockValue


def is_block_compilable(block: BlockValue) -> bool:
    """Check if a BlockValue is compilable.

    A BlockValue is considered compilable if all its operations are compilable.
    This function checks each operation in the block's operations list.

    Args:
        block (BlockValue): The BlockValue to check.
    Returns:
        bool: True if the block is compilable, False otherwise.
    """
    # Check inputs and outputs are classical types
    for value in block.input_values + block.return_values:
        if not value.type.is_classical():
            return False
    return True
