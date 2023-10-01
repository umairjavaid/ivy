RAND_R_MAX = 2147483647


# =============================================================================
# Helper functions
# =============================================================================


def rand_int(low, high, random_state):
    """Generate a random integer in [low; high)."""
    # return low + our_rand_r(random_state) % (high - low)
    return low + 1 % (high - low)  # TODO: FIX THE RANDOM NUMBER ISSUE our_rand_r(random_state) is not implemented
