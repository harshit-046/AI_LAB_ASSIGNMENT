import re
import heapq
import string

class plagiarism_checker:
    """Plagiarism Detector system using A* algorithm"""
    
    def __init__(self, similarity_threshold=0.3):
        self.similarity_threshold = similarity_threshold
        self.path_taken = []
        self.matched_sentences = []
    
    def split_into_sentences(self, document):
        """Break document into sentences and clean them"""
        # Split on sentence endings
        raw_sentences = re.split(r'[.!?]+', document)
        
        cleaned = []
        for sent in raw_sentences:
            sent = sent.strip()
            if sent:
                # Make lowercase and remove punctuation
                sent = sent.lower()
                sent = ' '.join(sent.split())
                sent = sent.translate(str.maketrans('', '', string.punctuation))
                cleaned.append(sent)
        
        return cleaned
    
    def calculate_edit_distance(self, str1, str2):
        """Calculate how different two strings are using edit operations"""
        m = len(str1)
        n = len(str2)
        
        # Handle empty strings
        if m == 0:
            return n
        if n == 0:
            return m
        
        # Build DP table
        prev_row = list(range(n + 1))
        
        for i in range(m):
            curr_row = [i + 1]
            for j in range(n):
                if str1[i] == str2[j]:
                    cost = prev_row[j]
                else:
                    cost = min(prev_row[j] + 1,      # substitution
                             curr_row[j] + 1,         # insertion
                             prev_row[j + 1] + 1)     # deletion
                curr_row.append(cost)
            prev_row = curr_row
        
        return prev_row[n]
    
    def get_similarity_score(self, str1, str2):
        """Get normalized similarity between 0 and 1"""
        if not str1 and not str2:
            return 0.0
        
        max_length = max(len(str1), len(str2))
        if max_length == 0:
            return 0.0
        
        distance = self.calculate_edit_distance(str1, str2)
        return distance / max_length
    
    def estimate_remaining_cost(self, pos1, pos2, doc1_sents, doc2_sents):
        """Heuristic to estimate cost to goal state"""
        remaining_doc1 = len(doc1_sents) - pos1
        remaining_doc2 = len(doc2_sents) - pos2
        
        # Return absolute difference as admissible heuristic
        return abs(remaining_doc1 - remaining_doc2)
    
    def find_best_alignment(self, doc1_sentences, doc2_sentences):
        """Use A* to find optimal sentence alignment between documents"""
        
        # Initialize starting state: (g_cost, position_doc1, position_doc2, path_history)
        initial_state = (0, 0, 0, [])
        
        # Priority queue stores: (f_score, tie_breaker, state)
        open_list = [(0, 0, initial_state)]
        counter = 0
        
        # Track visited states with their costs
        visited_states = {}
        
        while open_list:
            current_f, _, current_state = heapq.heappop(open_list)
            g_cost, i, j, path = current_state
            
            # Check if goal reached
            if i == len(doc1_sentences) and j == len(doc2_sentences):
                self.path_taken = path
                return g_cost, path
            
            # Skip if already visited with better cost
            state_key = (i, j)
            if state_key in visited_states and visited_states[state_key] <= g_cost:
                continue
            
            visited_states[state_key] = g_cost
            
            # Generate next possible states
            
            # Option 1: Match current sentences from both docs
            if i < len(doc1_sentences) and j < len(doc2_sentences):
                match_cost = self.get_similarity_score(doc1_sentences[i], doc2_sentences[j])
                new_g = g_cost + match_cost
                new_h = self.estimate_remaining_cost(i + 1, j + 1, doc1_sentences, doc2_sentences)
                new_f = new_g + new_h
                new_path = path + [(i, j, 'match')]
                new_state = (new_g, i + 1, j + 1, new_path)
                
                counter += 1
                heapq.heappush(open_list, (new_f, counter, new_state))
            
            # Option 2: Skip sentence from first document
            if i < len(doc1_sentences):
                skip_penalty = 1.0
                new_g = g_cost + skip_penalty
                new_h = self.estimate_remaining_cost(i + 1, j, doc1_sentences, doc2_sentences)
                new_f = new_g + new_h
                new_path = path + [(i, -1, 'skip_first')]
                new_state = (new_g, i + 1, j, new_path)
                
                counter += 1
                heapq.heappush(open_list, (new_f, counter, new_state))
            
            # Option 3: Skip sentence from second document
            if j < len(doc2_sentences):
                skip_penalty = 1.0
                new_g = g_cost + skip_penalty
                new_h = self.estimate_remaining_cost(i, j + 1, doc1_sentences, doc2_sentences)
                new_f = new_g + new_h
                new_path = path + [(-1, j, 'skip_second')]
                new_state = (new_g, i, j + 1, new_path)
                
                counter += 1
                heapq.heappush(open_list, (new_f, counter, new_state))
        
        return float('inf'), []
    
    def check_plagiarism(self, document1, document2):
        """Main function to detect plagiarism between two documents"""
        
        # Step 1: Preprocess documents
        sents1 = self.split_into_sentences(document1)
        sents2 = self.split_into_sentences(document2)
        
        # Step 2: Run A* alignment
        total_cost, alignment = self.find_best_alignment(sents1, sents2)
        
        # Step 3: Identify plagiarized content
        suspicious_pairs = []
        num_matches = 0
        
        for step in alignment:
            if step[2] == 'match':
                idx1, idx2 = step[0], step[1]
                diff_score = self.get_similarity_score(sents1[idx1], sents2[idx2])
                
                num_matches += 1
                
                # If similarity is high (distance is low), flag as plagiarism
                if diff_score <= self.similarity_threshold:
                    similarity_percent = 1.0 - diff_score
                    suspicious_pairs.append({
                        'index_doc1': idx1,
                        'index_doc2': idx2,
                        'sentence_doc1': sents1[idx1],
                        'sentence_doc2': sents2[idx2],
                        'similarity_percentage': similarity_percent,
                        'difference_score': diff_score
                    })
        
        self.matched_sentences = suspicious_pairs
        
        # Step 4: Calculate overall statistics
        max_sents = max(len(sents1), len(sents2))
        plagiarism_percentage = len(suspicious_pairs) / max_sents if max_sents > 0 else 0
        
        output = {
            'alignment_cost': total_cost,
            'sentences_in_doc1': len(sents1),
            'sentences_in_doc2': len(sents2),
            'total_matches': num_matches,
            'suspicious_matches': suspicious_pairs,
            'num_suspicious': len(suspicious_pairs),
            'plagiarism_percentage': plagiarism_percentage,
            'has_plagiarism': len(suspicious_pairs) > 0
        }
        
        return output
    
    def display_results(self, test_num, test_name, results):
        """Display the plagiarism detection results"""
        print(f"TEST - {test_num:02d} {test_name.upper()}")
        print()

        print(f"first document has {results['sentences_in_doc1']} sentences")
        print(f"second document has {results['sentences_in_doc2']} sentences")
        print()
        
        print("RESULT")
        print(f"{'suspicious matches'} {results['num_suspicious']}")
        print(f"{'plagiarism percentage'} {results['plagiarism_percentage']*100:.2f}%")
        print()

        if results['has_plagiarism']:
            print(f"{'final result'} PLAGIARISM DETECTED")
        else:
            print(f"{'final result'} NO PLAGIARISM DETECTED")
        
        print()


def run_tests():
    """Execute test cases for the system"""
    
    # Test Case 1: Identical Documents
    text1 = """
    Artificial intelligence is transforming the world. Machine learning algorithms 
    are becoming more sophisticated. Deep learning has revolutionized computer vision.
    """
    
    text2 = """
    Artificial intelligence is transforming the world. Machine learning algorithms 
    are becoming more sophisticated. Deep learning has revolutionized computer vision.
    """
    
    detector = plagiarism_checker(similarity_threshold=0.3)
    result = detector.check_plagiarism(text1, text2)
    detector.display_results(1, "Identical Documents", result)
    
    # Test Case 2: Slightly Modified Document
    text1 = """
    Artificial intelligence is transforming the world. Machine learning algorithms 
    are becoming more sophisticated. Deep learning has revolutionized computer vision.
    """
    
    text2 = """
    AI is changing the world. Machine learning models are getting more advanced. 
    Deep learning has transformed computer vision systems.
    """
    
    detector = plagiarism_checker(similarity_threshold=0.3)
    result = detector.check_plagiarism(text1, text2)
    detector.display_results(2, "Slightly Modified Document", result)
    
    # Test Case 3: Completely Different Documents
    text1 = """
    The weather today is sunny and warm. I enjoy going to the beach on weekends. 
    Swimming is my favorite activity in summer.
    """
    
    text2 = """
    Quantum computing uses qubits instead of bits. Superposition allows multiple states. 
    Quantum entanglement enables faster computation.
    """
    
    detector = plagiarism_checker(similarity_threshold=0.3)
    result = detector.check_plagiarism(text1, text2)
    detector.display_results(3, "Completely Different Documents", result)
    
    # Test Case 4: Partial Overlap
    text1 = """
    Climate change is a global challenge. Rising temperatures affect ecosystems. 
    Renewable energy is the solution. Solar and wind power are sustainable.
    """
    
    text2 = """
    The economy is growing steadily. Rising temperatures affect ecosystems. 
    Investment in technology is crucial. Solar and wind power are sustainable.
    """
    
    detector = plagiarism_checker(similarity_threshold=0.3)
    result = detector.check_plagiarism(text1, text2)
    detector.display_results(4, "Partial Overlap", result)

if __name__ == "__main__":
    run_tests()