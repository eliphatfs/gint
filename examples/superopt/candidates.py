"""Candidate generation for superoptimization."""

import numpy as np
from .opcodes import SearchOp, OP_LOAD_1D, OP_STORE_1D, OP_HALT, OP_NOP, MAX_STACK


def enumerate_exact_length(ops, input_depth, output_depth, length):
    """Enumerate all stack-valid instruction sequences of exactly *length*.

    Yields lists of SearchOp.  Uses depth-reachability pruning to cut the
    search tree aggressively.
    """
    if not ops:
        return
    max_inc = max((op.net_effect for op in ops), default=0)
    max_dec = max((-op.net_effect for op in ops), default=0)

    def _recurse(depth, pos, seq):
        if pos == length:
            if depth == output_depth:
                yield list(seq)
            return
        remaining = length - pos - 1          # instructions still to place after this one
        for op in ops:
            if depth < op.min_depth:
                continue
            nd = depth + op.net_effect
            if nd < 0 or nd > MAX_STACK:
                continue
            # Reachability: can we still hit output_depth from nd in *remaining* steps?
            if remaining == 0:
                if nd != output_depth:
                    continue
            else:
                lo = nd - remaining * max_dec
                hi = nd + remaining * max_inc
                if output_depth < lo or output_depth > hi:
                    continue
            seq.append(op)
            yield from _recurse(nd, pos + 1, seq)
            seq.pop()

    yield from _recurse(input_depth, 0, [])


def enumerate_exact_length_indices(ops, input_depth, output_depth, length):
    """Enumerate all stack-valid sequences of *length*, emitting op-indices.

    Functionally equivalent to enumerate_exact_length but returns a
    contiguous int32 numpy array of shape (N, length) — column j is the
    index into `ops` chosen at body position j. Skips the per-leaf
    Python list of SearchOp objects (a major cost at 14M leaves) by
    writing directly into a preallocated buffer.
    """
    if not ops or length == 0:
        return np.empty((0, length), dtype=np.int32)
    max_inc = max((op.net_effect for op in ops), default=0)
    max_dec = max((-op.net_effect for op in ops), default=0)

    # Cache (min_depth, net_effect, idx) triples — avoids attribute lookups
    # inside the hot recursion (each op is touched billions of times).
    op_table = [(op.min_depth, op.net_effect, i) for i, op in enumerate(ops)]

    # Preallocate per-row scratch + grow result buffer geometrically
    # (avoids Python list-of-rows + final np.stack).
    cur = np.empty(length, dtype=np.int32)
    cap = 1 << 12
    buf = np.empty((cap, length), dtype=np.int32)
    n_out = 0

    def _recurse(depth, pos):
        nonlocal cap, buf, n_out
        if pos == length:
            if depth == output_depth:
                if n_out == cap:
                    cap *= 2
                    new_buf = np.empty((cap, length), dtype=np.int32)
                    new_buf[:n_out] = buf[:n_out]
                    buf = new_buf
                buf[n_out] = cur
                n_out += 1
            return
        remaining = length - pos - 1
        for min_depth, net_effect, idx in op_table:
            if depth < min_depth:
                continue
            nd = depth + net_effect
            if nd < 0 or nd > MAX_STACK:
                continue
            if remaining == 0:
                if nd != output_depth:
                    continue
            else:
                lo = nd - remaining * max_dec
                hi = nd + remaining * max_inc
                if output_depth < lo or output_depth > hi:
                    continue
            cur[pos] = idx
            _recurse(nd, pos + 1)

    _recurse(input_depth, 0)
    return buf[:n_out]


def make_prefix(arity):
    """Build the load-input bytecode prefix (flat list of int32 words).

    Loads *arity* input tensors from slots 0..arity-1.
    """
    prefix = []
    for i in range(arity):
        prefix.extend([OP_LOAD_1D, i])   # fldg_1d(offset=0, arg_i=i)
    return prefix


def make_suffix(arity):
    """Build the store-output + halt suffix.

    Stores result into tensor slot *arity* (the first output slot).
    """
    return [OP_STORE_1D, arity, OP_HALT, 0]


def sequences_to_bytecodes(sequences, arity):
    """Convert a list of SearchOp sequences into a numpy bytecode array.

    All sequences must have the same length.
    Returns: numpy (N, total_words) int32 array — complete programs ready for GPU.
    """
    if not sequences:
        return np.empty((0, 0), dtype=np.int32)
    body_len = len(sequences[0])

    prefix = make_prefix(arity)
    suffix = make_suffix(arity)
    prefix_words = len(prefix)
    body_words = body_len * 2
    suffix_words = len(suffix)
    total_words = prefix_words + body_words + suffix_words

    result = np.empty((len(sequences), total_words), dtype=np.int32)

    # Broadcast prefix and suffix
    for j, v in enumerate(prefix):
        result[:, j] = v
    base = prefix_words + body_words
    for j, v in enumerate(suffix):
        result[:, base + j] = v

    # Fill bodies
    body_start = prefix_words
    for i, seq in enumerate(sequences):
        for j, op in enumerate(seq):
            result[i, body_start + j * 2]     = op.opcode
            result[i, body_start + j * 2 + 1] = op.operand

    return result


def make_reference_bytecode(body, arity):
    """Build a single complete program from a body of (opcode, operand) pairs."""
    prefix = make_prefix(arity)
    suffix = make_suffix(arity)
    words = prefix[:]
    for op, operand in body:
        words.extend([op, operand])
    words.extend(suffix)
    return np.array(words, dtype=np.int32)


def random_valid_batch(ops, input_depth, output_depth, length, n, rng=None):
    """Generate *n* random stack-valid sequences of *length* using numpy.

    For each position, sample uniformly from the valid ops at the current depth.
    Sequences that can't reach output_depth are retried (rejection).
    Returns a list of lists of SearchOp (only valid sequences).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_ops = len(ops)
    min_depths = np.array([op.min_depth for op in ops])
    net_effects = np.array([op.net_effect for op in ops])
    max_inc = max(net_effects)
    max_dec = -min(net_effects)

    # (n, length) array of op indices
    result_indices = np.empty((n, length), dtype=np.int32)
    depths = np.full(n, input_depth, dtype=np.int32)
    alive = np.ones(n, dtype=bool)

    for pos in range(length):
        remaining = length - pos - 1
        # valid[i, j] = can candidate i use ops[j] at this position?
        valid = (depths[:, None] >= min_depths[None, :])
        new_d = depths[:, None] + net_effects[None, :]
        valid &= (new_d >= 0) & (new_d <= MAX_STACK)
        if remaining == 0:
            valid &= (new_d == output_depth)
        else:
            lo = new_d - remaining * max_dec
            hi = new_d + remaining * max_inc
            valid &= (output_depth >= lo) & (output_depth <= hi)

        # Uniform sampling from valid ops per candidate
        probs = valid.astype(np.float64)
        row_sums = probs.sum(axis=1, keepdims=True)
        dead = row_sums.ravel() == 0
        alive &= ~dead
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs /= row_sums
        cum = np.cumsum(probs, axis=1)
        r = rng.random((n, 1))
        choices = (r < cum).argmax(axis=1).astype(np.int32)

        result_indices[:, pos] = choices
        depths = depths + net_effects[choices]

    # Filter to alive & correct final depth
    good = alive & (depths == output_depth)
    good_indices = result_indices[good]

    # Convert to lists of SearchOp
    out = []
    for row in good_indices:
        out.append([ops[j] for j in row])
    return out
