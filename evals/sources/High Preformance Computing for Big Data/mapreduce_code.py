"""Small MapReduce assignment fixture."""


def map_partition(records):
    """Partition records by key before the reduce phase."""
    return [(record[0], record[1]) for record in records]


def reduce_counts(pairs):
    """Aggregate counts for each partition key."""
    totals = {}
    for key, value in pairs:
        totals[key] = totals.get(key, 0) + value
    return totals
