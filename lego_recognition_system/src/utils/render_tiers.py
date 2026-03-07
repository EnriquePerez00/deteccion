def calculate_mix_params(num_unique_pieces, mix_ratio=0.75):
    """
    Calculate optimal render parameters based on the Tier Model for N (1-100+).
    T: Total Images
    M: Pieces per Image
    D: Different piece types per image
    """
    N = num_unique_pieces
    
    if N <= 5:
        # Tier 1: Micro-Set
        T, M, D = 650, 8, 2
    elif N <= 20:
        # Tier 2: Small-Set
        T, M, D = 1600, 12, 5
    elif N <= 50:
        # Tier 3: Medium-Set
        T, M, D = 3250, 18, 10
    elif N <= 100:
        # Tier 4: Large-Set
        T, M, D = 5250, 22, 12
    else:
        # Tier 5: Ultra-Set
        T, M, D = 6000, 25, 15

    # Safety: D cannot exceed N
    D = min(D, N)
    
    return T, D, M
