def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    res = {}
    for s in sentences:
       for w in s:
           if w not in res:
               res[w] = 0

           res[w] += 1
    return res