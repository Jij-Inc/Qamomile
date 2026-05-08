from qamomile.circuit.ir.block import Block


def is_block_compilable(block: Block) -> bool:
    """Check if a Block is compilable.

    A Block is considered compilable if all its operations are compilable.
    This function checks each operation in the block's operations list.

    Args:
        block (Block): The Block to check.
    Returns:
        bool: True if the block is compilable, False otherwise.
    """
    # Check inputs and outputs are classical types
    for value in block.input_values + block.output_values:
        if not value.type.is_classical():
            return False
    return True
