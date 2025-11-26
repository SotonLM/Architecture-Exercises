def quality_score(prompt, text):
    words = text.strip().split()
    numWords = len(words)

    lengthScore = 0
    repetitionScore = 0
    punctuationScore = 0
    relevanceScore = 0

    if numWords == 0:
        return 0.0
    
    if numWords < 5:
        lengthScore = 0.2  
    elif numWords > 70:
        lengthScore = 0.3
    elif 25 <= numWords <= 40:
        lengthScore = 1.0
    else:
        lengthScore = 0.5

    uniqueWords = set(word.lower().strip(".,!?;:") for word in words)
    uniquenessRatio = len(uniqueWords) / numWords 

    if uniquenessRatio < 0.4:
        repetitionScore = 0.2 
    elif uniquenessRatio < 0.7:
        repetitionScore = 0.6
    elif uniquenessRatio < 1:
        repetitionScore = 0.8
    else:
        repetitionScore = 1.0

    end_char = text.strip()[-1]
    if end_char in [".", "!", "?"]:
        punctuationScore = 1.0
    else:
        punctuationScore = 0.3

    promptWords = set(word.lower().strip(".,!?;:") for word in prompt.split())
    overlap = promptWords.intersection(uniqueWords)

    if not promptWords:
        relevanceScore = 0.5 
    else:
        overlapRatio = len(overlap) / len(promptWords)
        if overlapRatio == 0:
            relevanceScore = 0.1
        elif overlapRatio < 0.5:
            relevanceScore = 0.6
        elif overlapRatio < 1:
            relevanceScore = 0.8
        else:
            relevanceScore = 1.0


    total = 0.2*lengthScore + 0.3*repetitionScore + 0.1*punctuationScore + 0.4*relevanceScore

    return (0.2*lengthScore*100, 0.3*repetitionScore*100, 0.1*punctuationScore*100, 0.4*relevanceScore*100, float(round(total * 100, 1)))
