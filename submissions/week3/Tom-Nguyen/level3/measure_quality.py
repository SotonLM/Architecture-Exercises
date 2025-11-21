import re
from collections import Counter

class QualityMeasure:
    def __init__(self):
        self.weights = {
            'fluency': 0.30,
            'coherence': 0.30,
            'relevance': 0.40
        }
    def evaluate(self, text, prompt):
        score_fluency = self.score_fluency(text) 
        score_coherence = self.score_coherence(text)
        score_relevace = self.score_relevance(text,prompt)
        total_score = (
            score_fluency * self.weights['fluency'] + 
            score_coherence * self.weights['coherence'] +
            score_relevace * self.weights['relevance']
        )
        return total_score
        
    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def tokenize(self, text):
        text_lower = text.lower()
        words = re.findall(r'[a-z]+', text_lower)
        return [w for w in words if len(w) > 2]

    def score_fluency(self, text: str) -> float:
            score = 100.0
            
            sentences = self.split_sentences(text)
            if not sentences:
                return 0
            capitalized = sum(1 for s in sentences if s and s[0].isupper())
            cap_ratio = capitalized / len(sentences)
            score *= cap_ratio
    
            proper_ending = 0 
            for s in sentences:
                if s and s[-1] in '.!?':
                    proper_ending += 1
            punct_ratio = proper_ending / len(sentences)
            score *= punct_ratio

            
            words = text.split()
            avg_sentence_length = len(words) / len(sentences)
            if avg_sentence_length < 5:
                score *= 0.7
        

            word_list = [w.lower() for w in words if len(w) > 3]
            if word_list:
                word_freq = Counter(word_list)
                max_freq = max(word_freq.values())
                repetition_ratio = max_freq / len(word_list)
                if repetition_ratio > 0.2:  
                    score *= (1 - repetition_ratio)
  
            return max(0, min(100, score))
        
    def score_coherence(self, text: str) -> int:
        score = 100
        sentences = self.split_sentences(text)

        if len(sentences) < 2:
            return 50
        transition_words = {'however', 'therefore', 'moreover', 'consequently', 'furthermore', 'nevertheless', 'additionally', 'similarly', 'on the other hand', 'in contrast', 'for example', 'for instance', 'in conclusion'}

        transition_used = sum(1 for s in sentences if any(tw in s.lower() for tw in transition_words))
        transition_score = min(100, (transition_used / len(sentences)) * 100)

        
        cohesion_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(self.tokenize(sentences[i]))
            words2 = set(self.tokenize(sentences[i + 1]))
            
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                cohesion_scores.append(overlap)
        
        avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
        cohesion_score = min(100, avg_cohesion * 200)  

        score = (transition_score * 0.4 + cohesion_score * 0.6)
        return max(0, min(100, score))
    def score_relevance(self, text: str, prompt: str) -> float:
    
        prompt_words = set(self.tokenize(prompt))
        text_words = set(self.tokenize(text))
        
        if not prompt_words:
            return 50  
        
        overlap = len(prompt_words & text_words)
        union = len(prompt_words | text_words)
        
        if union == 0:
            return 0
        
        jaccard_score = (overlap / union) * 100
        
        key_words_found = overlap / len(prompt_words) * 100
        
        score = (key_words_found * 0.6) + (jaccard_score * 0.4)
        
        return min(100, score)
    

if __name__ == "__main__":
    test = [
    {
        'id': 1,
        'prompt': 'Explain the benefits of exercise for mental health.',
        'response': 'Regular exercise has profound benefits for mental health. Physical activity releases endorphins, which are natural mood elevators that can reduce symptoms of depression and anxiety. Moreover, exercise improves sleep quality, which is crucial for emotional regulation. Studies have shown that individuals who engage in regular physical activity report higher levels of self-esteem and cognitive function. Additionally, exercise provides a healthy coping mechanism for stress and can foster social connections when done in group settings.',
        'label': 'High Quality - On Topic'
    },
    {
        'id': 2,
        'prompt': 'Explain the benefits of exercise for mental health.',
        'response': 'Exercise is good. Exercise is very good. People should exercise. Exercise helps you. It makes you feel better. Exercise is important.',
        'label': 'Low Quality - Repetitive & Vague'
    },
    {
        'id': 3,
        'prompt': 'Describe how photosynthesis works in plants.',
        'response': 'Photosynthesis is the process by which plants convert light energy into chemical energy. During this process, plants absorb carbon dioxide from the air and water from the soil. Chlorophyll in the leaves captures sunlight, which drives the chemical reactions that produce glucose and oxygen. The glucose serves as food for the plant, while oxygen is released as a byproduct. This remarkable process occurs primarily in the chloroplasts of leaf cells and is essential for life on Earth.',
        'label': 'High Quality - Accurate & Complete'
    },
    {
        'id': 4,
        'prompt': 'Describe how photosynthesis works in plants.',
        'response': 'I really love gardening! My favorite flowers are roses and tulips. Yesterday I planted some tomatoes in my backyard. The weather has been beautiful lately. Gardening is such a relaxing hobby.',
        'label': 'Low Quality - Off Topic'
    },
    {
        'id': 5,
        'prompt': 'What are the main causes of World War I?',
        'response': 'World War I was caused by a complex web of factors. The primary causes included militarism, where European powers engaged in an arms race building up their military capabilities. Alliance systems created a situation where a conflict between two nations could quickly escalate. Imperialism led to competition over colonies and resources. Nationalism fueled ethnic tensions, particularly in the Balkans. The assassination of Archduke Franz Ferdinand served as the immediate trigger, but these underlying tensions made war almost inevitable.',
        'label': 'High Quality - Comprehensive'
    },
    {
        'id': 6,
        'prompt': 'What are the main causes of World War I?',
        'response': 'There was fighting. Countries were mad. Then war started. It was bad.',
        'label': 'Low Quality - Underdeveloped'
    },
    {
        'id': 7,
        'prompt': 'Explain the difference between renewable and non-renewable energy.',
        'response': 'Renewable energy sources, such as solar, wind, and hydroelectric power, can be replenished naturally and are virtually inexhaustible. These sources produce minimal environmental pollution and greenhouse gas emissions. In contrast, non-renewable energy sources like coal, oil, and natural gas are finite resources formed over millions of years. Once depleted, they cannot be replaced within a human timeframe. Non-renewable sources also contribute significantly to air pollution and climate change when burned.',
        'label': 'High Quality - Clear Contrast'
    },
    {
        'id': 8,
        'prompt': 'Explain the difference between renewable and non-renewable energy.',
        'response': 'Energy is like super important for everything we do ya know. Like we need it for cars and houses and stuff. Some energy is good and some is like not so good I guess. Solar panels are cool.',
        'label': 'Medium Quality - Informal & Incomplete'
    },
    {
        'id': 9,
        'prompt': 'What is machine learning and how is it used?',
        'response': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves algorithms that identify patterns and make predictions based on training data. Machine learning is used in numerous applications including recommendation systems, fraud detection, medical diagnosis, and autonomous vehicles. For example, streaming services use machine learning to suggest content based on viewing history, while banks employ it to detect unusual transaction patterns that might indicate fraud.',
        'label': 'High Quality - Well Explained'
    },
    {
        'id': 10,
        'prompt': 'What is machine learning and how is it used?',
        'response': 'Machine learning machine learning is when machines learn. They learn things. Machines use learning to learn. This is called machine learning because machines learn.',
        'label': 'Low Quality - Circular & Repetitive'
    }
]
    new_measure = QualityMeasure()
    for item in test:
        score = new_measure.evaluate(item['response'], item['prompt'])
        print(f"ID: {item['id']}, Label: {item['label']}, Score: {score:.2f}")
    
