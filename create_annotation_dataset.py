"""
Vote Identification Dataset Creation

This script processes deliberation data from MongoDB and creates an annotated dataset
for token classification to identify votes, voters, and voting outcomes.

Annotation scheme:
- C.1. Votação (Vote)
- C.2. Votante (Voter)
- C.2. Contabilização global (Global count - unanimidade/maioria)
- Ligação_objeto de votação (Link between vote and subject)
- Ligação_posicionamento (Link between voter and vote position)
  - a favor (in favor)
  - contra (against)
  - abstenção (abstention)
- Ligação_resultado (Link between global count and vote)
  - por unanimidade (unanimously)
  - por maioria (by majority)
"""

import os
import re
import json
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv
from collections import defaultdict
from pymongo import MongoClient
import spacy
import logging
from tqdm import tqdm
import random
import unicodedata
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional, Set, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('../../.env')

# Function to remove accents from text
def remove_accents(text):
    """Remove accents and special characters from text."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if not unicodedata.combining(c)])

# Constants for BIO tagging scheme
TAG_MAPPING = {
    "O": 0,                            # Outside of any entity
    "B-VOTACAO": 1,                    # Beginning of voting action
    "I-VOTACAO": 2,                    # Inside of voting action
    "B-VOTANTE": 3,                    # Beginning of voter
    "I-VOTANTE": 4,                    # Inside of voter
    "B-CONTABILIZACAO": 5,             # Beginning of global counting
    "I-CONTABILIZACAO": 6,             # Inside of global counting
    "B-LIGACAO_POSICIONAMENTO_FAVOR": 7,    # Beginning of favorable position link
    "I-LIGACAO_POSICIONAMENTO_FAVOR": 8,   # Inside of favorable position link
    "B-LIGACAO_POSICIONAMENTO_CONTRA": 9,  # Beginning of against position link
    "I-LIGACAO_POSICIONAMENTO_CONTRA": 10,  # Inside of against position link
    "B-LIGACAO_POSICIONAMENTO_ABSTENCAO": 11,  # Beginning of abstention position link
    "I-LIGACAO_POSICIONAMENTO_ABSTENCAO": 12,  # Inside of abstention position link
    "B-LIGACAO_RESULTADO_UNANIMIDADE": 13,  # Beginning of unanimous result link
    "I-LIGACAO_RESULTADO_UNANIMIDADE": 14,  # Inside of unanimous result link
    "B-LIGACAO_RESULTADO_MAIORIA": 15,      # Beginning of majority result link
    "I-LIGACAO_RESULTADO_MAIORIA": 16,      # Inside of majority result link
}

# Reverse mapping for convenience
ID_TO_TAG = {v: k for k, v in TAG_MAPPING.items()}

# Load Portuguese language model
try:
    nlp = spacy.load("pt_core_news_lg")
    logger.info("Loaded Portuguese language model")
except OSError:
    logger.warning("Portuguese model not found. Installing...")
    os.system("python -m spacy download pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")


class VoteAnnotationDatasetCreator:
    """Creates an annotated dataset for vote identification from MongoDB data."""

    def __init__(self, mongo_uri=None, db_name="citilink", 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """Initialize with MongoDB connection details and dataset split ratios."""
        self.mongo_uri = mongo_uri
        
        # Set dataset split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow slight floating point imprecision
            logger.warning(f"Dataset split ratios should sum to 1.0, but got {total_ratio}. Adjusting ratios.")
            # Normalize to ensure they sum to 1
            self.train_ratio = train_ratio / total_ratio
            self.val_ratio = val_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
        
        if not self.mongo_uri:
            for env_var in ['MONGO_URI', 'MONGODB_URI', 'DB_URI']:
                if os.getenv(env_var):
                    self.mongo_uri = os.getenv(env_var)
                    logger.info(f"Found MongoDB URI in environment variable {env_var}")
                    break
            
            parent_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if not self.mongo_uri and os.path.exists(parent_env_path):
                logger.info(f"Found .env file at {parent_env_path}, trying to load it")
                load_dotenv(parent_env_path)
                self.mongo_uri = os.getenv('MONGO_URI')
        
        if not self.mongo_uri:
            logger.warning("MongoDB URI not found in environment variables, using default local connection")
            self.mongo_uri = "mongodb://localhost:27017/"
        
        self.db_name = db_name
        self.client = None
        self.db = None
        self.participants_dict = {}
        self.participants_party_dict = {}
        
        self.vote_keywords = [
            r'\bvotar\w*\b', r'\bvotação\b', r'\bvotaç[õo]es\b',
            r'\bdecid\w+\b', r'\baprova\w*\b', r'\brejeita\w*\b',
            r'\bdelibera\w*\b', r'\baprovado\b', r'\baprovada\b'
        ]
        
        self.position_keywords = {
            'favor': [r'\ba favor\b', r'\bfavorável\b', r'\bfavoráveis\b', r'\bconcord\w+\b'],
            'contra': [r'\bcontra\b', r'\bdesfavorável\b', r'\bdesfavoráveis\b', r'\bdiscord\w+\b', r'\bnão concord\w+\b'],
            'abstencao': [r'\babsten\w+\b', r'\babstiv\w+\b', r'\babsté\w+\b']
        }
        
        self.result_keywords = {
            'unanimidade': [r'\bunanim\w+\b', r'\btodos\b', r'\bpor todos\b'],
            'maioria': [r'\bmaioria\b', r'\bmaior parte\b', r'\bparcial\w+\b']
        }

        # Enhanced patterns for party references with gender specificity
        self.party_reference_patterns = {
            'female_singular': [
                r'a vereadora d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'a eleita pel[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'a deputada d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'a representante d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)'
            ],
            'female_plural': [
                r'as vereadoras d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'as eleitas pel[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'as deputadas d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'as representantes d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)'
            ],
            'male_singular': [
                r'o vereador d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'o eleito pel[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'o deputado d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'o representante d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)'
            ],
            'male_plural': [
                r'os vereadores d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'os eleitos pel[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'os deputados d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)',
                r'os representantes d[oa]s? ([A-Za-zçãõáéíóúâêôà\s-]+)'
            ]
        }

    def connect_to_db(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            return True
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            return False

    def load_participants(self):
        """
        Load participant data from MongoDB.
        
        Returns:
            Dict mapping participant IDs to names
        """
        if not self.client:
            if not self.connect_to_db():
                return {}
                
        try:
            collection = self.db["participante"]
            participants_cursor = collection.find({})
            
            # Create mapping from ID to name
            participants_dict = {}
            self.participants_party_dict = {}
            
            for participant in participants_cursor:
                participant_id = str(participant['_id'])
                name = participant.get('name', 'Unknown')
                participants_dict[participant_id] = name
                
                # Store party information if available
                if 'party' in participant:
                    self.participants_party_dict[participant_id] = participant['party']
            
            logger.info(f"Loaded {len(participants_dict)} participants")
            return participants_dict
        except Exception as e:
            logger.error(f"Error loading participants: {str(e)}")
            return {}

    def load_vereador_data(self):
        """
        Load vereador (city council member) data from MongoDB.
        Maps clean names (without accents) to party information.
        
        Returns:
            Dict mapping clean vereador names to their complete information
        """
        if not self.client:
            if not self.connect_to_db():
                return {}
                
        try:
            collection = self.db["vereador"]
            vereadores_cursor = collection.find({})
            
            # Create mapping from cleaned name to vereador info with party
            vereador_dict = {}
            
            for vereador in vereadores_cursor:
                name = vereador.get('name', '')
                if not name:
                    continue
                
                # Clean the name by removing accents for matching
                clean_name = remove_accents(name).lower()
                
                vereador_dict[clean_name] = {
                    'nome': name,
                    'partido': vereador.get('party', ''),
                    'camara': vereador.get('municipio', '')
                }
            
            logger.info(f"Loaded {len(vereador_dict)} vereadores from database")
            return vereador_dict
        except Exception as e:
            logger.error(f"Error loading vereadores: {str(e)}")
            return {}
    
    def load_vereador_data_from_csv(self):
        """
        Load vereador (city council member) data from CSV file.
        Used as fallback when MongoDB data is not available.
        
        Returns:
            Dict mapping clean vereador names to their complete information
        """
        try:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vareadores.csv')
            if not os.path.exists(csv_path):
                logger.error(f"Vereador CSV file not found at {csv_path}")
                return {}
                
            df = pd.read_csv(csv_path, encoding='utf-8')
            vereador_dict = {}
            
            for _, row in df.iterrows():
                name = row.get('nome', '')
                if not name:
                    continue
                
                # Clean the name by removing accents for matching
                clean_name = remove_accents(name).lower()
                
                vereador_dict[clean_name] = {
                    'nome': name,
                    'partido': row.get('partido', ''),
                    'camara': row.get('municipio', '')
                }
            
            logger.info(f"Loaded {len(vereador_dict)} vereadores from CSV file")
            return vereador_dict
        except Exception as e:
            logger.error(f"Error loading vereadores from CSV: {str(e)}")
            return {}
    
    def split_dataset(self, dataset, output_dir):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            dataset: List of dataset examples
            output_dir: Directory to save the split datasets
            
        Returns:
            Dict containing the split datasets
        """
        try:
            # Check if we have a dataset to split
            if not dataset:
                # Try loading from full_dataset.jsonl if available
                full_dataset_path = os.path.join(output_dir, 'full_dataset.jsonl')
                if os.path.exists(full_dataset_path):
                    logger.info(f"Loading existing dataset from {full_dataset_path}")
                    dataset = []
                    with open(full_dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                example = json.loads(line.strip())
                                dataset.append(example)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in dataset file, skipping line")
                else:
                    logger.error("No dataset provided and no full_dataset.jsonl found")
                    return {}
            
            logger.info(f"Splitting dataset with {len(dataset)} examples")
            
            # Use stratified sampling to ensure similar distribution of tag types
            # Create a simple feature for stratification based on what tags are present
            example_features = []
            for example in dataset:
                unique_tags = set(tag for tag in example['tags'] if tag != "O")
                # Simple feature: number of unique tag types
                feature = len(unique_tags)
                example_features.append(feature)
            
            # Check if we have enough examples for stratification
            use_stratification = len(set(example_features)) > 1
            
            # Count occurrences of each feature value
            feature_counts = {}
            for f in example_features:
                if f not in feature_counts:
                    feature_counts[f] = 0
                feature_counts[f] += 1
            
            # Check if any feature has only 1 occurrence
            for f, count in feature_counts.items():
                if count < 2:
                    use_stratification = False
                    logger.warning(f"Feature value {f} has only {count} occurrence(s), disabling stratification")
                    break
            
            # First split into train and temp sets
            train_indices, temp_indices = train_test_split(
                range(len(dataset)),
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.random_seed,
                stratify=example_features if use_stratification else None
            )
            
            # Further split temp into validation and test sets
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            
            # Extract features for the temp set
            temp_features = [example_features[i] for i in temp_indices]
            
            # Check if the temp set is suitable for stratified splitting
            use_temp_stratification = len(set(temp_features)) > 1
            
            # Count occurrences of each feature value in the temp set
            temp_feature_counts = {}
            for f in temp_features:
                if f not in temp_feature_counts:
                    temp_feature_counts[f] = 0
                temp_feature_counts[f] += 1
            
            # Check if any feature has only 1 occurrence in the temp set
            for f, count in temp_feature_counts.items():
                if count < 2:
                    use_temp_stratification = False
                    logger.warning(f"Feature value {f} has only {count} occurrence(s) in temp set, disabling stratification for val/test split")
                    break
            
            # Split temp into validation and test
            val_indices_from_temp, test_indices_from_temp = train_test_split(
                range(len(temp_indices)),
                test_size=(1 - val_ratio_adjusted),
                random_state=self.random_seed,
                stratify=temp_features if use_temp_stratification else None
            )
            
            # Convert back to original indices
            val_indices = [temp_indices[i] for i in val_indices_from_temp]
            test_indices = [temp_indices[i] for i in test_indices_from_temp]
            
            # Create the split datasets
            train_dataset = [dataset[i] for i in train_indices]
            val_dataset = [dataset[i] for i in val_indices]
            test_dataset = [dataset[i] for i in test_indices]
            
            logger.info(f"Split dataset into: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
            
            # Save each split to a file
            def save_split(split_name, split_data):
                split_file = os.path.join(output_dir, f'{split_name}_dataset.jsonl')
                with open(split_file, 'w', encoding='utf-8') as f:
                    for example in split_data:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                logger.info(f"Saved {len(split_data)} examples to {split_file}")
                return split_file
            
            train_file = save_split('train', train_dataset)
            val_file = save_split('val', val_dataset)
            test_file = save_split('test', test_dataset)
            
            # Also save as huggingface format for easy loading
            self.save_huggingface_format(train_dataset, os.path.join(output_dir, 'train.json'))
            self.save_huggingface_format(val_dataset, os.path.join(output_dir, 'val.json'))
            self.save_huggingface_format(test_dataset, os.path.join(output_dir, 'test.json'))
            
            return {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset,
                'train_file': train_file,
                'val_file': val_file,
                'test_file': test_file
            }
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            logger.exception("Stack trace:")
            return {}
    
    def save_huggingface_format(self, dataset, output_file):
        """
        Save dataset in HuggingFace format.
        
        Args:
            dataset: List of dataset examples
            output_file: File path to save the HF format
        """
        try:
            # Convert to HuggingFace format
            hf_format = []
            for example in dataset:
                # Skip examples with mismatch between tokens and tags
                if len(example['tokens']) != len(example['tags']):
                    logger.warning(f"Example {example['id']} has mismatched tokens ({len(example['tokens'])}) and tags ({len(example['tags'])}). Skipping.")
                    continue
                    
                hf_example = {
                    "id": example['id'],
                    "tokens": example['tokens'],
                    "ner_tags": example['tag_ids']
                }
                hf_format.append(hf_example)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(hf_format, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(hf_format)} examples in HuggingFace format to {output_file}")
        except Exception as e:
            logger.error(f"Error saving HuggingFace format: {str(e)}")
    
    def save_tags_to_jsonl(self, dataset, output_dir):
        """
        Save all unique tags to a separate JSONL file.
        
        Args:
            dataset: List of dataset examples
            output_dir: Directory to save the tags file
            
        Returns:
            Path to the saved file
        """
        try:
            # Extract all unique tags from the dataset
            unique_tags = set()
            for example in dataset:
                if 'tags' in example:
                    unique_tags.update(tag for tag in example['tags'] if tag != "O")
            
            # Sort tags for consistency
            sorted_tags = sorted(unique_tags)
            
            # Create a list of tag entries with their ids
            tag_entries = []
            for tag in sorted_tags:
                tag_entry = {
                    "tag": tag,
                    "tag_id": TAG_MAPPING.get(tag, 0)
                }
                tag_entries.append(tag_entry)
            
            # Add the "Outside" tag too
            tag_entries.insert(0, {"tag": "O", "tag_id": TAG_MAPPING["O"]})
            
            # Save to JSONL file
            tags_file = os.path.join(output_dir, 'tags.jsonl')
            with open(tags_file, 'w', encoding='utf-8') as f:
                for entry in tag_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(tag_entries)} tags to {tags_file}")
            return tags_file
        except Exception as e:
            logger.error(f"Error saving tags to JSONL: {str(e)}")
            return None

    def get_participant_info(self, participant_id):
        """
        Get information about a participant directly from the database.
        
        Args:
            participant_id: MongoDB ObjectID of the participant
            
        Returns:
            Dict with participant information or None if not found
        """
        if not self.client:
            if not self.connect_to_db():
                return None
                
        try:
            collection = self.db["participante"]
            participant = collection.find_one({"_id": participant_id})
            
            if participant:
                return {
                    "name": participant.get("name", "Unknown"),
                    "party": participant.get("party", "")
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving participant info: {str(e)}")
            return None
    
    def find_matching_vereador_party(self, clean_participant_name, vereadores):
        """
        Find the party of a vereador that matches the given participant name.
        
        Args:
            clean_participant_name: Clean participant name (no accents, lowercase)
            vereadores: Dict mapping clean vereador names to their info
            
        Returns:
            Party name or empty string if no match found
        """
        # Try exact match first
        if clean_participant_name in vereadores:
            return vereadores[clean_participant_name]['partido']
        
        # Try partial matches
        best_match = None
        highest_score = 0
        
        for vereador_name, info in vereadores.items():
            # Try matching on full name, first name, last name
            vereador_parts = vereador_name.split()
            participant_parts = clean_participant_name.split()
            
            # Check if first and last names match
            if len(vereador_parts) > 0 and len(participant_parts) > 0:
                if vereador_parts[0] == participant_parts[0]:
                    # First names match
                    score = 1
                    if len(vereador_parts) > 1 and len(participant_parts) > 1:
                        if vereador_parts[-1] == participant_parts[-1]:
                            # Last names also match
                            score = 2
                    
                    if score > highest_score:
                        highest_score = score
                        best_match = info
            
            # Check for substring match
            if vereador_name in clean_participant_name or clean_participant_name in vereador_name:
                score = len(set(vereador_name.split()) & set(clean_participant_name.split()))
                if score > highest_score:
                    highest_score = score
                    best_match = info
        
        return best_match['partido'] if best_match else ""

    def map_participants_to_vereadores(self, participants, vereadores):
        """
        Maps participants to vereadores using clean text matching.
        
        Args:
            participants: Dict mapping participant IDs to names
            vereadores: Dict mapping clean names to vereador info
            
        Returns:
            Dict mapping participant IDs to party information
        """
        participant_party_mapping = {}
        
        # Create a reverse lookup dictionary for quick case-insensitive matching
        vereador_names = {name.lower(): info for name, info in vereadores.items()}
        
        for participant_id, participant_name in participants.items():
            if not participant_name:
                continue
                
            # Clean the participant name for matching
            clean_participant = remove_accents(participant_name).lower()
            
            # Try exact match first
            if clean_participant in vereador_names:
                info = vereador_names[clean_participant]
                participant_party_mapping[participant_id] = info['partido']
                continue
                
            # Try partial matches if no exact match found
            best_match = None
            highest_score = 0
            
            for vereador_name, info in vereadores.items():
                # Try matching on full name, first name, last name
                vereador_parts = vereador_name.split()
                participant_parts = clean_participant.split()
                
                # Check if first and last names match
                if len(vereador_parts) > 0 and len(participant_parts) > 0:
                    if vereador_parts[0] == participant_parts[0]:
                        # First names match
                        score = 1
                        if len(vereador_parts) > 1 and len(participant_parts) > 1:
                            if vereador_parts[-1] == participant_parts[-1]:
                                # Last names also match
                                score = 2
                        
                        if score > highest_score:
                            highest_score = score
                            best_match = info
                
                # Check for substring match
                if vereador_name in clean_participant or clean_participant in vereador_name:
                    score = len(set(vereador_name.split()) & set(clean_participant.split()))
                    if score > highest_score:
                        highest_score = score
                        best_match = info
            
            if best_match:
                participant_party_mapping[participant_id] = best_match['partido']
        
        logger.info(f"Mapped {len(participant_party_mapping)} participants to vereadores with party information")
        return participant_party_mapping

    def load_deliberations_with_votes(self):
        """
        Load deliberations that contain votes.     
        Returns:
            List of deliberation documents with votes
        """
        if not self.client:
            if not self.connect_to_db():
                return []
                
        try:
            collection = self.db["assunto"]
            # Find documents that have votes
            query = {"votos": {"$exists": True, "$not": {"$size": 0}}}
            projection = {
                "deliberacao": 1, 
                "votos": 1, 
                "assunto": 1,
                "title": 1,
                "_id": 1
            }
            
            deliberations_cursor = collection.find(query, projection).limit(1000)  # Added limit for debugging
            deliberations = []
            
            for delib in deliberations_cursor:
                # Filter only documents with actual deliberation text
                if delib.get('deliberacao') and isinstance(delib['deliberacao'], str):
                    # Add assunto_id for reference
                    delib['assunto_id'] = str(delib['_id'])
                    deliberations.append(delib)
            
            logger.info(f"Loaded {len(deliberations)} deliberations with votes (limited to 1000 for debugging)")
            return deliberations
        except Exception as e:
            logger.error(f"Error loading deliberations: {str(e)}")
            return []
            
    def _find_entities_in_text(self, text, participants, votes):
        """
        Find entities like votes, voters, and positions in text.
        
        Args:
            text: The deliberation text
            participants: Dict mapping participant IDs to names
            votes: List of vote dictionaries
            
        Returns:
            Dict of entity lists
        """
        doc = nlp(text)
        
        entities = {
            'votes': [],
            'voters': [],
            'positions': {
                'favor': [],
                'contra': [],
                'abstencao': []
            },
            'results': {
                'unanimidade': [],
                'maioria': []
            }
        }
        
        # Find vote keywords
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check for vote actions
            for pattern in self.vote_keywords:
                if re.search(pattern, sent_text, re.IGNORECASE):
                    entities['votes'].append(sent.text)
                    break
            
            # Check for positions
            for position, patterns in self.position_keywords.items():
                for pattern in patterns:
                    if re.search(pattern, sent_text, re.IGNORECASE):
                        entities['positions'][position].append(sent.text)
                        break
            
            # Check for global results
            for result, patterns in self.result_keywords.items():
                for pattern in patterns:
                    if re.search(pattern, sent_text, re.IGNORECASE):
                        entities['results'][result].append(sent.text)
                        break
        
        # Find voters in text
        for vote in votes:
            participant_id = vote.get('participante')
            if not participant_id:
                continue
                
            participant_name = participants.get(str(participant_id), None)
            if not participant_name:
                continue
            
            # Look for voter name in text
            name_parts = participant_name.split()
            if len(name_parts) > 1:
                first_name = name_parts[0]
                last_name = name_parts[-1]
                
                # Try different name variations
                for name_var in [participant_name, first_name, last_name]:
                    if name_var.lower() in text.lower():
                        entities['voters'].append(name_var)
                        break
        
        return entities

    def _identify_party_vote_spans(self, text):
        """
        Identify spans in text where collective party votes are mentioned.
        
        Args:
            text: The deliberation text
            
        Returns:
            List of tuples (start_pos, end_pos, party, vote_type, full_match)
        """
        doc = nlp(text)
        party_vote_spans = []
        
        party_vote_patterns = [
            # Original patterns - group reference "do partido"
            r'(os vereadores|os deputados|os membros|a bancada) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+vot\w+\s+(\w+)',
            r'(os vereadores|os deputados|os membros|a bancada) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+)',
            
            # New patterns - "eleitos pelo" format
            r'(os eleitos|as eleitas|o eleito|a eleita) pel[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+vot\w+\s+(\w+)',
            r'(os eleitos|as eleitas|o eleito|a eleita) pel[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+)',
            
            # More generic party references
            r'(os representantes|as representantes|o representante|a representante) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+|vot\w+\s+\w+)',
            r'(a bancada) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+|vot\w+\s+\w+)',
            
            # Direct party vote patterns without group references
            r'([A-Z][A-Z\-]+)(?:\s+votou|votaram)\s+(a favor|contra|abst\w+)',
            r'[Oo]\s+(?:partido|grupo)\s+([A-Z][A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+vot\w+\s+(a favor|contra|abst\w+)',
            
            # Additional formats for elected members
            r'(os eleitos|as eleitas) (?:na|pela) lista d[ao]\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+|vot\w+\s+\w+)'
        ]
        
        for sent in doc.sents:
            for pattern in party_vote_patterns:
                for match in re.finditer(pattern, sent.text, re.IGNORECASE):
                    # Handle the case where the pattern might have different group structures
                    if pattern.startswith(r'([A-Z][A-Z\-]+)'):
                        # Direct party vote pattern
                        party = match.group(1).strip()
                        vote_position = match.group(2).strip().lower()
                        party_ref = party  # The party itself is the reference
                    elif pattern.startswith(r'[Oo]\s+(?:partido|grupo)'):
                        # Pattern for "O partido X votou"
                        party = match.group(1).strip()
                        vote_position = match.group(2).strip().lower()
                        party_ref = match.group(0)[:match.start(1)].strip()  # "O partido"
                    else:
                        # Standard pattern with group reference
                        party_ref = match.group(1).strip()  # "os vereadores", "os eleitos", etc.
                        party = match.group(2).strip()
                        vote_position = match.group(3).strip().lower()
                    
                    vote_type = "unknown"
                    if any(term in vote_position for term in ["favor", "favorável"]):
                        vote_type = "favor"
                    elif any(term in vote_position for term in ["contra", "desfavorável"]):
                        vote_type = "contra"
                    elif any(term in vote_position for term in ["abst", "abstenção"]):
                        vote_type = "abstencao"
                    
                    # Get the char span of the party reference
                    start_pos = match.start()
                    
                    # End position depends on the pattern type
                    if pattern.startswith(r'([A-Z][A-Z\-]+)'):
                        end_pos = match.start(2)  # End before the voting verb/position
                    elif pattern.startswith(r'[Oo]\s+(?:partido|grupo)'):
                        end_pos = match.start(2)  # End before the voting verb/position
                    else:
                        end_pos = match.start(3)  # Standard pattern
                    
                    party_vote_spans.append((
                        start_pos, 
                        end_pos, 
                        party, 
                        vote_type, 
                        match.group(0)
                    ))
        
        return party_vote_spans

    def _extract_gendered_party_references(self, text, party_to_participants):
        """
        Extract gendered party references from text and map them to participants.
        
        Args:
            text (str): The deliberation text
            party_to_participants (dict): Mapping from party name to list of participants
        
        Returns:
            List of tuples containing (sentence, reference_type, mentioned_party, matched_party, participants)
        """
        if not text or not party_to_participants:
            return []
        
        # Skip extremely long texts to prevent memory issues
        if len(text) > 50000:
            logger.warning("Text too long, truncating to 50000 chars for gendered reference extraction")
            text = text[:50000]
        
        doc = nlp(text)
        gendered_references = []
        
        # Process each sentence
        for sent in doc.sents:
            sent_text = sent.text
            
            # Process each gender reference pattern
            for ref_type, patterns in self.party_reference_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sent_text, re.IGNORECASE)
                    
                    for match in matches:
                        mentioned_party = match.group(1).strip()
                        clean_party = remove_accents(mentioned_party).lower()
                        
                        # Try to match with known parties
                        matched_party = None
                        filtered_participants = []
                        
                        # Find the best match for the mentioned party
                        best_match_score = 0
                        for party_name in party_to_participants.keys():
                            clean_known_party = remove_accents(party_name).lower()
                            
                            # Check for exact match
                            if clean_party == clean_known_party:
                                matched_party = party_name
                                filtered_participants = party_to_participants[party_name]
                                break
                            
                            # Check for partial match (party name contained in mention)
                            if clean_known_party in clean_party or clean_party in clean_known_party:
                                match_score = len(clean_known_party) / max(len(clean_party), 1)
                                if match_score > best_match_score:
                                    best_match_score = match_score
                                    matched_party = party_name
                                    filtered_participants = party_to_participants[party_name]
                        
                        # If no good match found, skip
                        if not matched_party:
                            continue
                        
                        # Filter participants by gender if this is a gender-specific reference
                        if 'female' in ref_type and filtered_participants:
                            filtered_participants = [
                                p for p in filtered_participants 
                                if p.get('nome', '').strip().endswith('a') 
                                and not p.get('nome', '').strip().lower().endswith(('costa', 'rocha'))
                            ]
                        elif 'male' in ref_type and filtered_participants:
                            filtered_participants = [
                                p for p in filtered_participants 
                                if not p.get('nome', '').strip().endswith('a') 
                                or p.get('nome', '').strip().lower().endswith(('costa', 'rocha'))
                            ]
                        
                        # Only add if we have participants
                        if filtered_participants:
                            gendered_references.append((
                                sent_text,
                                ref_type,
                                mentioned_party,
                                matched_party,
                                filtered_participants
                            ))
        
        return gendered_references

    def generate_token_annotations(self, text, participants, votes):
        """
        Generate BIO tag annotations for tokens in text.
        
        Args:
            text: The deliberation text
            participants: Dict mapping participant IDs to names
            votes: List of vote dictionaries
            
        Returns:
            List of (token, tag) tuples
        """
        doc = nlp(text)
        annotations = []
        
        # Get entities
        entities = self._find_entities_in_text(text, participants, votes)
        
        # Get collective party vote spans
        party_vote_spans = self._identify_party_vote_spans(text)
        
        # Iterate through tokens and assign tags
        for token in doc:
            token_text = token.text
            token_pos = token.idx
            token_end = token_pos + len(token_text)
            assigned_tag = "O"  # Default tag is Outside
            
            # Check if token is part of a collective party reference (e.g. "os vereadores do PS")
            for start_pos, end_pos, party, vote_type, full_match in party_vote_spans:
                if start_pos <= token_pos < end_pos:
                    if token_pos == start_pos:
                        assigned_tag = "B-VOTANTE"
                    else:
                        assigned_tag = "I-VOTANTE"
                    break
            
            # If not a party reference, check other entity types
            if assigned_tag == "O":
                # Check if token is part of a vote
                for vote_text in entities['votes']:
                    if token_text in vote_text:
                        if is_beginning_token(token, vote_text, text):
                            assigned_tag = "B-VOTACAO"
                        else:
                            assigned_tag = "I-VOTACAO"
                        break
            
            # Check if token is part of an individual voter name
            if assigned_tag == "O":
                for voter in entities['voters']:
                    if token_text in voter:
                        if is_beginning_token(token, voter, text):
                            assigned_tag = "B-VOTANTE"
                        else:
                            assigned_tag = "I-VOTANTE"
                        break
            
            # Check for position links
            if assigned_tag == "O":
                for position, texts in entities['positions'].items():
                    for pos_text in texts:
                        if token_text in pos_text:
                            tag_prefix = f"LIGACAO_POSICIONAMENTO_{position.upper()}"
                            if is_beginning_token(token, pos_text, text):
                                assigned_tag = f"B-{tag_prefix}"
                            else:
                                assigned_tag = f"I-{tag_prefix}"
                            break
                    if assigned_tag != "O":
                        break
            
            # Check for result links
            if assigned_tag == "O":
                for result, texts in entities['results'].items():
                    for res_text in texts:
                        if token_text in res_text:
                            tag_prefix = f"LIGACAO_RESULTADO_{result.upper()}"
                            if is_beginning_token(token, res_text, text):
                                assigned_tag = f"B-{tag_prefix}"
                            else:
                                assigned_tag = f"I-{tag_prefix}"
                            break
                    if assigned_tag != "O":
                        break
            
            annotations.append((token_text, assigned_tag))
        
        return annotations

    def create_dataset(self, output_dir='datasets'):
        """
        Create and save the annotated dataset, splitting into train/val/test sets.
        
        Args:
            output_dir: Directory to save the datasets
            
        Returns:
            Boolean indicating success
        """
        if not self.connect_to_db():
            return False
        
        # Load vereador data from MongoDB
        vereadores = self.load_vereador_data()
        
        if not vereadores:
            logger.warning("No vereador data found in MongoDB. Using CSV file instead.")
            vereadores = self.load_vereador_data_from_csv()
            if not vereadores:
                logger.error("Could not load vereador data from MongoDB or CSV")
                return False
        else:
            logger.info(f"Successfully loaded {len(vereadores)} vereadores from MongoDB")
        
        deliberations = self.load_deliberations_with_votes()
        
        if not deliberations:
            logger.error("Failed to load deliberation data")
            return False
        
        # Dictionary to store participant info as we encounter them
        participants_dict = {}
        participants_party_dict = {}
        
        dataset = []
        for delib in tqdm(deliberations, desc="Processing deliberations"):
            deliberation_text = delib['deliberacao']
            votes = delib['votos']
            
            # Process participants directly from the votes
            for vote in votes:
                participant_id = vote.get('participante')
                if not participant_id:
                    continue
                
                participant_id_str = str(participant_id)
                
                # If we haven't seen this participant before, get their info from the database
                if participant_id_str not in participants_dict:
                    participant_info = self.get_participant_info(participant_id)
                    if participant_info:
                        participants_dict[participant_id_str] = participant_info['name']
                        
                        # Map to vereador immediately
                        clean_name = remove_accents(participant_info['name']).lower()
                        party = self.find_matching_vereador_party(clean_name, vereadores)
                        
                        if party:
                            participants_party_dict[participant_id_str] = party
            
            # Store the updated dictionaries for use in this method
            self.participants_dict = participants_dict
            self.participants_party_dict = participants_party_dict
            
            # Continue with the rest of the processing...
            annotations = self.generate_token_annotations(deliberation_text, participants_dict, votes)
            
            doc = nlp(deliberation_text)
            entities = self._find_entities_in_text(deliberation_text, participants_dict, votes)
            
            # Identify collective party references
            party_vote_spans = self._identify_party_vote_spans(deliberation_text)
            
            example = {
                'id': delib['assunto_id'],
                'title': delib['title'],
                'text': deliberation_text,
                'tokens': [token for token, _ in annotations],
                'tags': [tag for _, tag in annotations],
                'tag_ids': [TAG_MAPPING.get(tag, 0) for _, tag in annotations],
                'votes': []
            }
            
            for vote in votes:
                participant_id = vote.get('participante')
                if participant_id:
                    participant_id_str = str(participant_id)
                    participant_name = participants_dict.get(participant_id_str, "Unknown")
                    participant_party = participants_party_dict.get(participant_id_str, "")
                else:
                    participant_name = "Unknown"
                    participant_party = ""
                
                example['votes'].append({
                    'participante_id': str(participant_id) if participant_id else None,
                    'participante_nome': participant_name,
                    'partido': participant_party,
                    'tipo': vote.get('tipo', 'unknown')
                })
            
            # Add collective party references
            party_collective_references = []
            for start_pos, end_pos, party, vote_type, full_match in party_vote_spans:
                party_ref_text = deliberation_text[start_pos:end_pos]
                party_collective_references.append({
                    'text': party_ref_text,
                    'partido': party,
                    'tipo': vote_type
                })
            
            example['party_collective_references'] = party_collective_references
            
            party_to_participants = defaultdict(list)
            for vote in example['votes']:
                if vote['partido']:
                    party_to_participants[vote['partido']].append({
                        'id': vote['participante_id'],
                        'nome': vote['participante_nome'],
                        'tipo': vote['tipo']
                    })
            
            gendered_references = self._extract_gendered_party_references(deliberation_text, dict(party_to_participants))
            
            party_references = []
            for sentence, ref_type, mentioned_party, matched_party, filtered_participants in gendered_references:
                party_references.append({
                    'sentence': sentence,
                    'reference_type': ref_type,
                    'mentioned_party': mentioned_party, 
                    'matched_party': matched_party,
                    'participants': [{'id': p['id'], 'nome': p['nome']} for p in filtered_participants[:5]]
                })
            
            example['party_to_participants'] = dict(party_to_participants)
            example['gendered_party_references'] = party_references
            
            dataset.append(example)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete dataset for reference
        full_dataset_file = os.path.join(output_dir, 'full_dataset.jsonl')
        with open(full_dataset_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Full dataset saved to {full_dataset_file} with {len(dataset)} examples")
        
        # Split and save train/val/test datasets
        dataset_splits = self.split_dataset(dataset, output_dir)
        
        # Save tags to a separate JSONL file
        self.save_tags_to_jsonl(dataset, output_dir)
        
        # Log dataset statistics for the full dataset
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total examples: {len(dataset)}")
        
        tag_counts = defaultdict(int)
        for example in dataset:
            for tag in example['tags']:
                tag_counts[tag] += 1
        
        logger.info(f"  Entity distribution:")
        for tag, count in sorted(tag_counts.items()):
            if tag != "O":
                logger.info(f"    {tag}: {count}")
        
        # Count collective party references instead of party_votes
        collective_ref_count = 0
        for example in dataset:
            collective_ref_count += len(example.get('party_collective_references', []))
        
        logger.info(f"  Total collective party references detected: {collective_ref_count}")
        
        participants_with_party = sum(1 for example in dataset 
                                      for vote in example['votes'] 
                                      if vote.get('partido'))
        
        logger.info(f"  Participants with party information: {participants_with_party}")
        
        gendered_ref_counts = defaultdict(int)
        participants_via_gendered_refs = 0
        
        for example in dataset:
            refs = example.get('gendered_party_references', [])
            for ref in refs:
                gendered_ref_counts[ref['reference_type']] += 1
                participants_via_gendered_refs += len(ref.get('participants', []))
        
        logger.info(f"  Gendered party references:")
        for ref_type, count in sorted(gendered_ref_counts.items()):
            logger.info(f"    {ref_type}: {count}")
        logger.info(f"  Participants identified via gendered references: {participants_via_gendered_refs}")
        
        return True


# Helper function for token annotation
def is_beginning_token(token, entity_text, full_text):
    """Determine if a token is at the beginning of an entity mention."""
    # Get the position of the entity in the full text
    entity_pos = full_text.find(entity_text)
    if entity_pos == -1:
        return False
        
    # Get the position of the token in the full text
    token_pos = token.idx
    
    # The token is at the beginning if its position matches the entity position
    return token_pos == entity_pos


if __name__ == "__main__":
    # Create the dataset creator with default train/val/test split ratios
    creator = VoteAnnotationDatasetCreator(
        train_ratio=0.7,  # 70% training
        val_ratio=0.15,   # 15% validation
        test_ratio=0.15,  # 15% test
        random_seed=42    # For reproducible splits
    )
    
    # Run the dataset creation process
    output_dir = 'datasets'
    success = creator.create_dataset(output_dir)
    
    if success:
        logger.info(f"Dataset creation completed successfully. Output saved to {output_dir} directory")
    else:
        logger.error("Dataset creation failed.")