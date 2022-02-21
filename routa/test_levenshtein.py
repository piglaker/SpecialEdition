

def iterative_levenshtein(string, target, costs=(1, 1, 1)):
    """

        piglaker modified version : 
            return edits
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """
    rows = len(string) + 1
    cols = len(target) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)] # dist = np.zeros(shape=(rows, cols))

    edits = [ [[] for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if string[row - 1] == target[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)  # substitution

            # record edit
            min_distance = dist[row][col]

            if min_distance == dist[row - 1][col] + deletes:
                edit = ("delete", string[row - 1])
                edits[row][col] = edits[row-1][col] + [edit]

            elif min_distance == dist[row][col - 1] + inserts:
                edit = ("insert", col, target[col-1])
                edits[row][col] = edits[row][col-1] + [edit]

            else:
                edit = ("substitution", string[row-1], target[col-1])
                edits[row][col] =  edits[row-1][col-1] + [edit]
            
    return dist[row][col], edits[row][col]


result, edits = iterative_levenshtein([1,2,3], [0,13])

print(result, edits)
