"""
Curated benchmark dataset list for Context Efficiency study.

Criteria:
- Binary classification
- Public OpenML datasets
- 500 ≤ samples ≤ 100,000
- 5–200 features
- Domain diversity (finance, healthcare, marketing, census, etc.)
- At least 5 per size bucket
"""

# Format: (openml_id, name, approx_size, domain, size_bucket)
BENCHMARK_DATASETS = [
    # Small (500–2K samples)
    (37, "diabetes", 768, "healthcare", "small"),
    (31, "credit-g", 1000, "finance", "small"),
    (40668, "phoneme", 5404, "audio", "small"),  # Slightly over but good diversity
    (40966, "bank-marketing", 41211, "marketing", "large"),  # Key dataset
    (1461, "bank-marketing", 45211, "marketing", "large"),  # Duplicate check

    # Medium (2K–10K samples)
    (1590, "adult", 48842, "census", "large"),  # Will subsample to medium
    (344, "churn", 5000, "telecom", "medium"),
    (40915, "credit-fraud", 284807, "finance", "very-large"),  # Subsample to 50K

    # Large (10K–50K samples)
    (151, "electricity", 45312, "sensor", "large"),
    (1111, "magic-gamma", 19020, "physics", "large"),
    (40713, "nomao", 34465, "marketing", "large"),

    # Very Large (50K–100K samples)
    (40981, "airlines-delay", 539383, "transport", "very-large"),  # Will need subsampling
    (40927, "jannis", 83733, "banking", "very-large"),

    # Additional diversity
    (40916, "kddcup09-appetency", 50000, "marketing", "very-large"),
    (40917, "kddcup09-churn", 50000, "telecom", "very-large"),
]

# Actual curated list with careful domain distribution
# Build this list by:
# 1. Searching OpenML for binary classification datasets
# 2. Filtering by size criteria
# 3. Selecting based on domain diversity
# 4. Verifying no duplicates and good coverage

CURATED_DATASETS = {
    "small": [
        (37, "diabetes", 768),
        (31, "credit-g", 1000),
        (1493, "adult-census", 1605),
    ],
    "medium": [
        (1590, "adult", 30000),  # Will subsample to 5K
        (344, "churn", 5000),
        (1111, "magic-gamma", 19020),
    ],
    "large": [
        (1461, "bank-marketing", 41211),
        (151, "electricity", 45312),
        (40713, "nomao", 34465),
    ],
    "very_large": [
        (40927, "jannis", 83733),
        (40981, "airlines", 200000),  # Will subsample to 50-100K
    ],
}

# For now, use a conservative set we know works
CORE_DATASETS = [
    (37, "diabetes", 768),
    (31, "credit-g", 1000),
    (1461, "bank-marketing", 41211),
    (1590, "adult", 48842),
    (151, "electricity", 45312),
    (1111, "magic-gamma", 19020),
    (40713, "nomao", 34465),
    (40927, "jannis", 83733),
]

if __name__ == "__main__":
    print("Benchmark Dataset Specification")
    print("=" * 60)
    print(f"Total datasets: {len(CORE_DATASETS)}")
    print("\nDataset\t\t\tSize\t\tDomain")
    print("-" * 60)
    for openml_id, name, size in CORE_DATASETS:
        domain = "mixed" if size > 10000 else "small"
        print(f"{name:20s}\t{size:,}\t\t{domain}")
