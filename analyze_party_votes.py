"""
Political Party Vote Analysis Tool

This script analyzes the vote identification dataset to visualize and explore
how political parties are mentioned and associated with voting patterns.

Compatible with limited dataset (100 assuntos)
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import re
from collections import Counter, defaultdict
import spacy
from colorama import Fore, Style, init
import os
import logging
import warnings
from pathlib import Path

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load Portuguese language model if visualization needs linguistic processing
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    logger.warning("Portuguese model not found. Installing...")
    os.system("python -m spacy download pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")


class PartyVoteAnalyzer:
    """Analyzes political party voting patterns in the dataset."""
    
    def __init__(self, dataset_file=None):
        """Initialize with dataset file path."""
        # If no dataset file is provided, look for it in common locations
        if dataset_file is None:
            possible_paths = [
                'datasets/full_dataset.jsonl',
                '../datasets/full_dataset.jsonl',
                '../../datasets/full_dataset.jsonl',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/full_dataset.jsonl'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/full_dataset.jsonl')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_file = path
                    break
                    
            if dataset_file is None:
                logger.warning("No dataset file found. Please specify the path to the dataset.")
                dataset_file = 'datasets/full_dataset.jsonl'  # Default as fallback
        
        self.dataset_file = dataset_file
        self.dataset = []
        self.load_dataset()
        
        # Patterns for identifying party references with gender and number
        self.party_reference_patterns = [
            # Female singular patterns
            r'a vereadora d[oa]s? (\w+)',
            r'a eleita pel[oa]s? (\w+)',
            r'a deputada d[oa]s? (\w+)',
            r'a representante d[oa]s? (\w+)',
            
            # Female plural patterns
            r'as vereadoras d[oa]s? (\w+)',
            r'as eleitas pel[oa]s? (\w+)',
            r'as deputadas d[oa]s? (\w+)',
            r'as representantes d[oa]s? (\w+)',
            
            # Male singular patterns
            r'o vereador d[oa]s? (\w+)',
            r'o eleito pel[oa]s? (\w+)',
            r'o deputado d[oa]s? (\w+)',
            r'o representante d[oa]s? (\w+)',
            
            # Male plural patterns
            r'os vereadores d[oa]s? (\w+)',
            r'os eleitos pel[oa]s? (\w+)',
            r'os deputados d[oa]s? (\w+)',
            r'os representantes d[oa]s? (\w+)',
            
            # Non-gendered patterns
            r'a bancada d[oa]s? (\w+)',
            r'o grupo d[oa]s? (\w+)',
            r'os membros d[oa]s? (\w+)',
            r'pel[oa]s? (\w+)'  # Generic "by the [PARTY]" pattern
        ]
    
    def load_dataset(self):
        """Load the dataset from file."""
        try:
            # Check if the dataset file exists
            if not os.path.exists(self.dataset_file):
                logger.error(f"Dataset file not found: {self.dataset_file}")
                logger.info(f"Looking for dataset in parent directories...")
                
                # Try to find the file in parent directories
                parent = Path(os.path.abspath(__file__)).parent
                for _ in range(3):  # Look up to 3 levels up
                    parent = parent.parent
                    dataset_path = parent / "datasets" / "full_dataset.jsonl"
                    if dataset_path.exists():
                        self.dataset_file = str(dataset_path)
                        logger.info(f"Found dataset at: {self.dataset_file}")
                        break
            
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.dataset.append(json.loads(line))
            logger.info(f"Loaded {len(self.dataset)} examples from dataset")
            
            if len(self.dataset) <= 100:
                logger.info(f"Working with a limited dataset ({len(self.dataset)} examples)")
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.dataset_file}")
            self.dataset = []
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON in dataset file")
            self.dataset = []
    
    def count_party_mentions(self):
        """Count mentions of political parties in the dataset with gender and number analysis."""
        party_counts = Counter()
        party_vote_counts = defaultdict(Counter)
        party_gender_counts = defaultdict(lambda: defaultdict(int))  # Changed to int instead of Counter
        
        for example in self.dataset:
            # Check party collective references recorded in the dataset
            for party_ref in example.get('party_collective_references', []):
                party = party_ref.get('partido')
                vote_type = party_ref.get('tipo')
                
                if party and vote_type:
                    party_counts[party] += 1
                    party_vote_counts[party][vote_type] += 1
            
            # Analyze text for gendered party references
            text = example.get('text', '')
            if text:
                # Find gendered references to parties
                party_refs = self._extract_party_references(text)
                for ref_type, party in party_refs:
                    party_gender_counts[party][ref_type] += 1
        
        logger.info(f"Found {len(party_counts)} distinct parties mentioned")
        return party_counts, party_vote_counts, party_gender_counts
    
    def _extract_party_references(self, text):
        """Extract party references with gender and number information."""
        # Skip extremely long texts to prevent memory issues
        if len(text) > 50000:
            logger.warning("Text too long, truncating to 50000 chars for party reference extraction")
            text = text[:50000]
            
        doc = nlp(text)
        references = []
        
        # Define reference types
        ref_types = {
            'female_singular': [
                r'a vereadora d[oa]s?', r'a eleita pel[oa]s?', 
                r'a deputada d[oa]s?', r'a representante d[oa]s?'
            ],
            'female_plural': [
                r'as vereadoras d[oa]s?', r'as eleitas pel[oa]s?',
                r'as deputadas d[oa]s?', r'as representantes d[oa]s?'
            ],
            'male_singular': [
                r'o vereador d[oa]s?', r'o eleito pel[oa]s?',
                r'o deputado d[oa]s?', r'o representante d[oa]s?'
            ],
            'male_plural': [
                r'os vereadores d[oa]s?', r'os eleitos pel[oa]s?',
                r'os deputados d[oa]s?', r'os representantes d[oa]s?'
            ],
            'neutral': [
                r'a bancada d[oa]s?', r'o grupo d[oa]s?',
                r'os membros d[oa]s?', r'pel[oa]s?'
            ]
        }
        
        # Common Portuguese political party acronyms (direct pattern)
        party_pattern = r'\b(PS|PSD|CDS-PP|BE|PCP|CDU|IL|CHEGA|PAN|Livre|PPM|MPT|Aliança|Nós,?\s?Cidadãos)\b'
        
        # First, directly search for party mentions
        party_matches = re.finditer(party_pattern, text, re.IGNORECASE)
        for match in party_matches:
            party_name = match.group(1).strip()
            references.append(('direct_mention', party_name))
        
        # Then check for gendered references
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check each reference type
            for ref_type, patterns in ref_types.items():
                for pattern in patterns:
                    # Fixed the invalid escape sequence by using a raw string
                    combined_pattern = f"{pattern} ([A-Za-zçãõáéíóúâêôà,\\s-]+)"
                    for match in re.finditer(combined_pattern, sent_text, re.IGNORECASE):
                        party_name = match.group(1).strip()
                        # Clean up party name (remove common trailing words)
                        party_name = re.sub(r'(?:,|\s+e|\s+que|\s+para|\s+com|\s+de|\s+do|\s+da).*$', '', party_name).strip()
                        references.append((ref_type, party_name))
        
        return references
    
    def visualize_party_mentions(self, top_n=10):
        """Visualize the most mentioned political parties with gender analysis."""
        party_counts, party_vote_counts, party_gender_counts = self.count_party_mentions()
        
        if not party_counts:
            logger.warning("No party mentions found in the dataset")
            print("No party mentions found in the dataset. Please check if the dataset was loaded correctly.")
            return
            
        # Adjust top_n based on dataset size
        if len(party_counts) < top_n:
            logger.info(f"Only {len(party_counts)} parties found, adjusting visualization")
            top_n = max(len(party_counts), 3)  # Show at least 3 parties if available
            
        # Get top N most mentioned parties
        top_parties = [party for party, _ in party_counts.most_common(top_n)]
        party_data = []
        
        # Prepare data for DataFrame
        for party in top_parties:
            votes = party_vote_counts[party]
            gender_refs = party_gender_counts[party]
            
            party_data.append({
                'party': party,
                'total': party_counts[party],
                'favor': votes.get('favor', 0),
                'contra': votes.get('contra', 0),
                'abstencao': votes.get('abstencao', 0),
                'unknown': votes.get('unknown', 0),
                'female_singular': gender_refs.get('female_singular', 0),
                'female_plural': gender_refs.get('female_plural', 0),
                'male_singular': gender_refs.get('male_singular', 0),
                'male_plural': gender_refs.get('male_plural', 0),
                'neutral': gender_refs.get('neutral', 0)
            })
        
        # Create DataFrame
        df = pd.DataFrame(party_data)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Bar chart of total mentions
        plt.subplot(3, 1, 1)
        sns.barplot(x='party', y='total', data=df, palette='viridis')
        plt.title(f'Top {len(top_parties)} Political Parties Mentioned in Deliberations (Limited Dataset)')
        plt.xlabel('Political Party')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45)
        
        # Stacked bar chart of vote types
        plt.subplot(3, 1, 2)
        vote_types = ['favor', 'contra', 'abstencao', 'unknown']
        vote_colors = ['green', 'red', 'blue', 'gray']
        bottom = np.zeros(len(df))
        
        for i, vote_type in enumerate(vote_types):
            plt.bar(df['party'], df[vote_type], bottom=bottom, label=vote_type, color=vote_colors[i])
            bottom += df[vote_type].values
        
        plt.title('Vote Types by Political Party (Limited Dataset)')
        plt.xlabel('Political Party')
        plt.ylabel('Number of Votes')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Stacked bar chart of gender/number references
        plt.subplot(3, 1, 3)
        gender_types = ['female_singular', 'female_plural', 'male_singular', 'male_plural', 'neutral']
        gender_colors = ['pink', 'magenta', 'lightblue', 'blue', 'gray']
        bottom = np.zeros(len(df))
        
        for i, gender_type in enumerate(gender_types):
            if gender_type in df.columns:  # Check if column exists
                plt.bar(df['party'], df[gender_type], bottom=bottom, 
                       label=gender_type.replace('_', ' ').title(), color=gender_colors[i])
                bottom += df[gender_type].values
        
        plt.title('Gender and Number References by Political Party (Limited Dataset)')
        plt.xlabel('Political Party')
        plt.ylabel('Number of References')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'party_vote_analysis.png')
        
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        # Show visualization immediately
        plt.show()
    
    def print_party_vote_examples(self, top_n=3):
        """Print examples of party votes from the dataset with gender analysis."""
        if not self.dataset:
            logger.warning("No data available to analyze")
            print("No data available to analyze. Please check if the dataset was loaded correctly.")
            return
            
        # Adjust for small dataset
        if len(self.dataset) < top_n:
            top_n = len(self.dataset)
            logger.info(f"Only {top_n} examples available in the dataset")
            
        examples_shown = 0
        
        print(f"\n{Fore.GREEN}===== Party Vote Examples with Gender Analysis (Limited Dataset) ====={Style.RESET_ALL}")
        
        for example in self.dataset:
            party_collective_refs = example.get('party_collective_references', [])
            party_to_participants = example.get('party_to_participants', {})
            
            if party_collective_refs:
                print(f"\n{Fore.BLUE}Deliberation ID: {example['id']}{Style.RESET_ALL}")
                print(f"Title: {example['title']}")
                
                # Find the sentences that mention parties and votes
                # Truncate very long texts
                text = example.get('text', '')
                if len(text) > 50000:
                    logger.warning(f"Text too long ({len(text)} chars), truncating to 50000 chars")
                    text = text[:50000] + "..."
                
                doc = nlp(text)
                party_vote_sentences = []
                gender_references = []
                
                # Common Portuguese political party acronyms - use the same pattern as in _extract_party_references
                party_pattern = r'\b(PS|PSD|CDS-PP|BE|PCP|CDU|IL|CHEGA|PAN|Livre|PPM|MPT|Aliança|Nós,?\s?Cidadãos)\b'
                
                for sent in doc.sents:
                    has_party = re.search(party_pattern, sent.text, re.IGNORECASE) is not None
                    has_vote = re.search(r'\bvot\w+\b|a favor|contra|abst\w+', sent.text.lower()) is not None
                    
                    if has_party:
                        party_vote_sentences.append(sent.text)
                    
                    # Extract gender references but limit processing
                    if len(sent.text) < 500:  # Only process reasonably sized sentences
                        refs = self._extract_party_references(sent.text)
                        for ref_type, party in refs:
                            gender_references.append((sent.text, ref_type, party))
                
                # Print relevant sentences
                if party_vote_sentences:
                    print(f"{Fore.YELLOW}Party-mentioning sentences:{Style.RESET_ALL}")
                    for i, sent in enumerate(party_vote_sentences[:3], 1):  # Show max 3 sentences
                        print(f"{i}. {sent}")
                
                # Print gender references if found
                if gender_references:
                    print(f"\n{Fore.CYAN}Gender-specific party references:{Style.RESET_ALL}")
                    for sentence, ref_type, party in gender_references[:3]:
                        print(f"- {ref_type}: '{party}' in sentence: \"{sentence}\"")
                
                # Print party collective references
                print(f"\n{Fore.MAGENTA}Detected collective party references:{Style.RESET_ALL}")
                for pcr in party_collective_refs:
                    print(f"  - {pcr['partido']}: {pcr['tipo']} - \"{pcr['text']}\"")
                
                # Print party to participants mapping
                if party_to_participants:
                    print(f"\n{Fore.GREEN}Party to participants mapping:{Style.RESET_ALL}")
                    for party, participants in party_to_participants.items():
                        # Safer gender detection
                        female_count = sum(1 for p in participants 
                                           if p.get('nome', '').strip().endswith('a')
                                           and not p.get('nome', '').strip().lower().endswith(('costa', 'rocha')))
                        male_count = len(participants) - female_count
                        print(f"  - {party}: {len(participants)} members ({female_count} female, {male_count} male)")
                        for i, p in enumerate(participants[:2]):  # Show max 2 participants per party
                            print(f"    - {p.get('nome', 'Unknown')}: {p.get('tipo', 'Unknown')}")
                        if len(participants) > 2:
                            print(f"    - ... and {len(participants)-2} more")
                
                print("-" * 80)
                examples_shown += 1
                
                if examples_shown >= top_n:
                    break
        
        if examples_shown == 0:
            print(f"{Fore.YELLOW}No examples with party votes found in the dataset.{Style.RESET_ALL}")
    
    def analyze_party_vote_patterns(self):
        """Analyze how often parties vote together or against each other."""
        if not self.dataset:
            logger.warning("No data available to analyze")
            print("No data available to analyze. Please check if the dataset was loaded correctly.")
            return
            
        # Count how many times each party votes for each position
        party_positions = defaultdict(Counter)
        deliberation_votes = defaultdict(dict)
        
        for example in self.dataset:
            delib_id = example['id']
            
            # Record each party's vote on this deliberation
            for party_ref in example.get('party_collective_references', []):
                party = party_ref.get('partido')
                vote_type = party_ref.get('tipo')
                
                if party and vote_type:
                    party_positions[party][vote_type] += 1
                    deliberation_votes[delib_id][party] = vote_type
        
        if not party_positions:
            print(f"{Fore.YELLOW}No party voting data found in the dataset.{Style.RESET_ALL}")
            return
        
        # Find deliberations where multiple parties voted
        agreement_matrix = defaultdict(lambda: defaultdict(Counter))
        
        for delib_id, votes in deliberation_votes.items():
            parties = list(votes.keys())
            
            # For each pair of parties that voted on this deliberation
            for i in range(len(parties)):
                for j in range(i+1, len(parties)):
                    party1, party2 = parties[i], parties[j]
                    vote1, vote2 = votes[party1], votes[party2]
                    
                    # Record if they agreed or disagreed
                    if vote1 == vote2:
                        agreement_matrix[party1][party2]['agree'] += 1
                        agreement_matrix[party2][party1]['agree'] += 1
                    else:
                        agreement_matrix[party1][party2]['disagree'] += 1
                        agreement_matrix[party2][party1]['disagree'] += 1
        
        # Print results
        print(f"\n{Fore.GREEN}===== Party Voting Patterns (Limited Dataset) ====={Style.RESET_ALL}")
        
        # 1. Print each party's voting distribution
        print(f"\n{Fore.BLUE}Party Voting Distributions:{Style.RESET_ALL}")
        for party, votes in party_positions.items():
            total = sum(votes.values())
            if total > 0:
                favor_pct = votes.get('favor', 0) / total * 100
                contra_pct = votes.get('contra', 0) / total * 100
                abst_pct = votes.get('abstencao', 0) / total * 100
                
                print(f"{party}: Total votes: {total}")
                print(f"  Favor: {votes.get('favor', 0)} ({favor_pct:.1f}%)")
                print(f"  Contra: {votes.get('contra', 0)} ({contra_pct:.1f}%)")
                print(f"  Abstention: {votes.get('abstencao', 0)} ({abst_pct:.1f}%)")
        
        # 2. Print agreement patterns between parties
        if agreement_matrix:
            print(f"\n{Fore.BLUE}Party Agreement Patterns:{Style.RESET_ALL}")
            for party1, others in agreement_matrix.items():
                for party2, counts in others.items():
                    total = counts['agree'] + counts['disagree']
                    if total > 0:
                        agree_pct = counts['agree'] / total * 100
                        print(f"{party1} and {party2}: Agree {counts['agree']}/{total} times ({agree_pct:.1f}%)")
        else:
            print(f"\n{Fore.YELLOW}No multi-party voting patterns found in this limited dataset.{Style.RESET_ALL}")
    
    def debug_dataset(self):
        """Debug function to inspect the dataset structure and party mentions."""
        print(f"\n{Fore.YELLOW}===== Dataset Debugging ====={Style.RESET_ALL}")
        print(f"Dataset file: {self.dataset_file}")
        print(f"Total examples: {len(self.dataset)}")
        
        # Count structure elements
        has_party_collective_refs = 0
        has_text = 0
        has_title = 0
        
        # Sample one example in detail
        if self.dataset:
            print("\nSample document structure:")
            keys = self.dataset[0].keys()
            print(f"Available keys: {', '.join(keys)}")
        
        # Count examples with relevant fields
        for i, example in enumerate(self.dataset):
            if example.get('party_collective_references'):
                has_party_collective_refs += 1
            if example.get('text'):
                has_text += 1
            if example.get('title'):
                has_title += 1
        
        print(f"\nExamples with party_collective_references: {has_party_collective_refs}/{len(self.dataset)}")
        print(f"Examples with text: {has_text}/{len(self.dataset)}")
        print(f"Examples with title: {has_title}/{len(self.dataset)}")
        
        # Look for party mentions in all texts
        party_pattern = r'\b(PS|PSD|CDS-PP|BE|PCP|CDU|IL|CHEGA|PAN|Livre|PPM|MPT|Aliança|Nós,?\s?Cidadãos)\b'
        total_party_mentions = 0
        documents_with_parties = 0
        party_occurrence = Counter()
        
        print(f"\n{Fore.CYAN}Scanning for party mentions in text...{Style.RESET_ALL}")
        
        for i, example in enumerate(self.dataset):
            text = example.get('text', '')
            if text:
                matches = re.findall(party_pattern, text, re.IGNORECASE)
                if matches:
                    documents_with_parties += 1
                    total_party_mentions += len(matches)
                    for match in matches:
                        party_occurrence[match.upper()] += 1
        
        print(f"Documents with party mentions: {documents_with_parties}/{len(self.dataset)}")
        print(f"Total party mentions found: {total_party_mentions}")
        
        if party_occurrence:
            print("\nParty occurrence in texts:")
            for party, count in party_occurrence.most_common():
                print(f"  {party}: {count} mentions")
        else:
            print("\nNo party mentions found in text. Expanding search patterns...")
            
            # Try broader patterns
            expanded_patterns = [
                r'\bpartido[s]?\b',  # "partido", "partidos"
                r'\bbancada[s]?\b',   # "bancada", "bancadas"
                r'\bvereador(es|a|as)?\b',  # "vereador", "vereadores", "vereadora", "vereadoras"
                r'\bdeputado(s|a|as)?\b'    # "deputado", "deputados", "deputada", "deputadas"
            ]
            
            for pattern in expanded_patterns:
                count = 0
                docs_with_match = 0
                for example in self.dataset:
                    text = example.get('text', '')
                    if text:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            docs_with_match += 1
                            count += len(matches)
                
                print(f"Pattern '{pattern}': found in {docs_with_match} documents, {count} total occurrences")
        
        # Check if party_collective_references field has expected structure
        if has_party_collective_refs > 0:
            print(f"\n{Fore.GREEN}Examples of party_collective_references structures:{Style.RESET_ALL}")
            examples_shown = 0
            for example in self.dataset:
                party_refs = example.get('party_collective_references', [])
                if party_refs:
                    print(f"\nDocument ID: {example.get('id', 'Unknown')}")
                    print(f"party_collective_references field ({len(party_refs)} items):")
                    for i, ref in enumerate(party_refs):
                        print(f"  {i+1}. {ref}")
                    
                    examples_shown += 1
                    if examples_shown >= 3:  # Show up to 3 examples
                        break
                        
        print(f"\n{Fore.YELLOW}===== End of Debugging ====={Style.RESET_ALL}")
        
        return {
            'total_examples': len(self.dataset),
            'with_party_collective_refs': has_party_collective_refs,
            'with_text': has_text,
            'with_party_mentions': documents_with_parties,
            'party_occurrence': party_occurrence
        }


if __name__ == "__main__":
    print(f"{Fore.GREEN}Political Party Vote Analysis Tool (for limited dataset){Style.RESET_ALL}")
    print("Running analysis on the dataset (limited to 100 assuntos)...\n")
    
    # Try to find the dataset file
    dataset_paths = [
        'datasets/full_dataset.jsonl',
        '../datasets/full_dataset.jsonl',
        '../../datasets/full_dataset.jsonl',
    ]
    
    found_dataset = None
    for path in dataset_paths:
        if os.path.exists(path):
            found_dataset = path
            print(f"Found dataset at: {path}")
            break
    
    analyzer = PartyVoteAnalyzer(dataset_file=found_dataset)
    
    if not analyzer.dataset:
        print(f"{Fore.RED}Error: Could not load any data from the dataset.{Style.RESET_ALL}")
        print("Please make sure the dataset exists and is properly formatted.")
    else:
        print(f"{Fore.GREEN}Successfully loaded {len(analyzer.dataset)} examples from the dataset.{Style.RESET_ALL}")
        
        # First run debugging to diagnose issues with party detection
        print(f"\n{Fore.YELLOW}Running dataset diagnostic to check for party mentions...{Style.RESET_ALL}")
        debug_results = analyzer.debug_dataset()
        
        # Only proceed with visualization if we found party mentions
        if debug_results['party_occurrence']:
            # Run all analyses
            analyzer.visualize_party_mentions()
            analyzer.print_party_vote_examples()
            analyzer.analyze_party_vote_patterns()
        else:
            print(f"\n{Fore.RED}No party mentions were detected in the dataset. Visual analysis skipped.{Style.RESET_ALL}")
            print("Possible reasons:")
            print("1. The dataset may not contain political party information.")
            print("2. The party naming conventions in the dataset differ from expected Portuguese party acronyms.")
            print("3. The dataset structure might not have the expected 'party_collective_references' field.")
            print("\nRecommended actions:")
            print("- Check a sample of the dataset content to verify the format.")
            print("- Adjust the party detection patterns in the code to match the format in your dataset.")
            print("- If your dataset uses different party naming conventions, add them to the party_pattern regex.")
        
        print(f"\n{Fore.GREEN}Analysis complete. Results are based on a limited dataset of {len(analyzer.dataset)} examples.{Style.RESET_ALL}")