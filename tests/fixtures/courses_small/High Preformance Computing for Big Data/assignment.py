def count_words(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        for word in line.split():
            counts[word] = counts.get(word, 0) + 1
    return counts
