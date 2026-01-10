def cs_alg(
    A: list[int], B: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Find common subarrays (matching segments) between two sequences.

    Returns only the matching segments where A and B have identical consecutive elements.
    These segments serve as anchor points - the gaps between them are diverging regions.

    Args:
        A: First token sequence
        B: Second token sequence

    Returns:
        Tuple of (segments_A, segments_B) where each is a list of (start, end) pairs.
        Each pair represents a matching segment where A[start:end] == B[start:end].
        Segments are non-overlapping and returned in sequential order.

    Example:
        A = [1, 2, 3, 4, 5]
        B = [1, 9, 8, 4, 5]
        Returns: ([(0,1), (3,5)], [(0,1), (3,5)])
        Meaning: A[0:1]=[1] matches B[0:1]=[1]
                 A[3:5]=[4,5] matches B[3:5]=[4,5]
                 (gaps between these are diverging regions)
    """
    from difflib import SequenceMatcher

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, A, B, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    segments_A, segments_B = [], []

    for a_start, b_start, size in matching_blocks:
        # Only add matching segments with non-zero length
        # The last block from get_matching_blocks() is always (len(A), len(B), 0)
        if size > 0:
            segments_A.append((a_start, a_start + size))
            segments_B.append((b_start, b_start + size))

    return segments_A, segments_B
