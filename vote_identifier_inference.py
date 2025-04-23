"""
Vote Identification Inference Module

This module provides inference capabilities for the vote identification model,
enabling detection of voters, vote positions, and mapping of collective party references
to individual participants when context is available.
"""

import os
import re
import json
import torch
import logging
import numpy as np
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import unicodedata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to remove accents
def remove_accents(text):
    """Remove accents and special characters from text."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if not unicodedata.combining(c)])


class VoteIdentifierInference:
    """Class for vote identification inference using a trained model."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the inference module.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Find the model path if not provided
        if model_path is None:
            possible_paths = [
                'vote_identifier_model/best_model',
                'src/vote_identification/vote_identifier_model/best_model',
                'src/data_generation_program/src/vote_identification/vote_identifier_model/best_model',
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path not found. Please provide a valid path to a trained model.")
        
        # Load tokenizer and model
        logger.info(f"Loading vote identification model from {model_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tag mapping
        tag_mapping_path = os.path.join(model_path, 'tag_mapping.json')
        if os.path.exists(tag_mapping_path):
            with open(tag_mapping_path, 'r', encoding='utf-8') as f:
                self.tag_mapping = json.load(f)
        else:
            # Try to find tag mapping in other locations
            possible_tag_paths = [
                'tag_mapping.json',
                'src/vote_identification/tag_mapping.json',
                'src/data_generation_program/src/vote_identification/tag_mapping.json',
            ]
            
            for path in possible_tag_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        self.tag_mapping = json.load(f)
                        break
            else:
                # Default tag mapping as fallback
                logger.warning("Tag mapping file not found. Using default mapping.")
                self.tag_mapping = {
                    "O": 0,                            # Outside of any entity
                    "B-VOTACAO": 1,                    # Beginning of voting action
                    "I-VOTACAO": 2,                    # Inside of voting action
                    "B-VOTANTE": 3,                    # Beginning of voter
                    "I-VOTANTE": 4,                    # Inside of voter
                    "B-PARTIDO": 5,                    # Beginning of political party
                    "I-PARTIDO": 6,                    # Inside of political party
                    "B-CONTABILIZACAO": 7,             # Beginning of global counting
                    "I-CONTABILIZACAO": 8,             # Inside of global counting
                    "B-LIGACAO_OBJETO": 9,             # Beginning of link between vote and subject
                    "I-LIGACAO_OBJETO": 10,            # Inside of link between vote and subject
                    "B-LIGACAO_POSICIONAMENTO_FAVOR": 11,    # Beginning of favorable position link
                    "I-LIGACAO_POSICIONAMENTO_FAVOR": 12,   # Inside of favorable position link
                    "B-LIGACAO_POSICIONAMENTO_CONTRA": 13,  # Beginning of against position link
                    "I-LIGACAO_POSICIONAMENTO_CONTRA": 14,  # Inside of against position link
                    "B-LIGACAO_POSICIONAMENTO_ABSTENCAO": 15,  # Beginning of abstention position link
                    "I-LIGACAO_POSICIONAMENTO_ABSTENCAO": 16,  # Inside of abstention position link
                    "B-LIGACAO_RESULTADO_UNANIMIDADE": 17,  # Beginning of unanimous result link
                    "I-LIGACAO_RESULTADO_UNANIMIDADE": 18,  # Inside of unanimous result link
                    "B-LIGACAO_RESULTADO_MAIORIA": 19,      # Beginning of majority result link
                    "I-LIGACAO_RESULTADO_MAIORIA": 20,      # Inside of majority result link
                }
        
        # Create inverse mapping
        self.id_to_tag = {int(v): k for k, v in self.tag_mapping.items()}
        
        # Define patterns for party references with gender specificity
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
        
        # Collective party vote patterns
        self.party_vote_patterns = [
            r'(os vereadores|os deputados|os membros|a bancada) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+vot\w+\s+(\w+)',
            r'(os vereadores|os deputados|os membros|a bancada) d[ao](?:\s+partido)?\s+([A-Z\-]+|[A-Za-zçãõáéíóúâêôà\s]+)\s+(a favor|contra|abst\w+)'
        ]
        
        logger.info(f"Vote identification model loaded and ready for inference on {self.device}")
    
    def predict(self, text, context=None):
        """
        Identify votes, voters, positions and outcomes in a text.
        
        Args:
            text (str): The deliberation text to analyze
            context (dict, optional): Context for mapping party references to participants.
                Expected format:
                {
                    'party_to_participants': {
                        'PARTY_NAME': [{'id': '123', 'nome': 'Person Name', 'tipo': 'favor|contra|abstencao'}],
                        ...
                    }
                }
        
        Returns:
            dict: Detected entities and information
        """
        # Input validation
        if not text or not isinstance(text, str):
            return {'error': 'Invalid input text'}
        
        # Tokenize text for the model
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Get offset mapping for mapping predictions back to original text
        offset_mapping = inputs.pop('offset_mapping').numpy()[0]
        
        # Move inputs to device
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0]
        
        # Map predictions to tags
        predicted_tags = [self.id_to_tag.get(p, 'O') for p in predictions]
        
        # Extract entities
        entities = self._extract_entities(text, predicted_tags, offset_mapping)
        
        # Identify collective party references
        party_references = self._identify_party_references(text)
        
        # Map party references to participants when context is available
        resolved_references = []
        if context and 'party_to_participants' in context:
            resolved_references = self._resolve_party_references(party_references, context['party_to_participants'])
        
        # Combine all results
        result = {
            'entities': entities,
            'party_references': party_references,
            'resolved_references': resolved_references
        }
        
        return result
    
    def _extract_entities(self, text, predicted_tags, offset_mapping):
        """
        Extract entities from the predicted tags.
        
        Args:
            text (str): The original text
            predicted_tags (list): Predicted BIO tags
            offset_mapping (list): Token offsets in the original text
        
        Returns:
            dict: Extracted entities by type
        """
        entities = {
            'votante': [],
            'votacao': [],
            'posicionamento': {
                'favor': [],
                'contra': [],
                'abstencao': []
            },
            'resultado': {
                'unanimidade': [],
                'maioria': []
            }
        }
        
        # Process each tag type
        current_entity = None
        current_type = None
        
        for i, (tag, offset) in enumerate(zip(predicted_tags, offset_mapping)):
            # Skip padding tokens
            if offset[0] == offset[1]:
                continue
            
            # Get the token's text
            token_text = text[offset[0]:offset[1]]
            
            # Process tag
            if tag.startswith('B-'):
                # End previous entity if any
                if current_entity:
                    self._add_entity_to_result(entities, current_type, current_entity)
                
                # Start a new entity
                current_entity = {
                    'text': token_text,
                    'start': offset[0],
                    'end': offset[1]
                }
                current_type = tag[2:]  # Remove 'B-' prefix
            
            elif tag.startswith('I-') and current_entity and tag[2:] == current_type:
                # Continue current entity
                current_entity['text'] += ' ' + token_text
                current_entity['end'] = offset[1]
            
            else:
                # End current entity or no entity
                if current_entity:
                    self._add_entity_to_result(entities, current_type, current_entity)
                    current_entity = None
                    current_type = None
        
        # Add the last entity if any
        if current_entity:
            self._add_entity_to_result(entities, current_type, current_entity)
        
        return entities
    
    def _add_entity_to_result(self, entities, entity_type, entity):
        """Add an entity to the appropriate category in the results."""
        if entity_type == 'VOTANTE':
            entities['votante'].append(entity)
        
        elif entity_type == 'VOTACAO':
            entities['votacao'].append(entity)
        
        elif entity_type.startswith('LIGACAO_POSICIONAMENTO_'):
            position_type = entity_type.split('_')[-1].lower()
            if position_type in entities['posicionamento']:
                entities['posicionamento'][position_type].append(entity)
        
        elif entity_type.startswith('LIGACAO_RESULTADO_'):
            result_type = entity_type.split('_')[-1].lower()
            if result_type in entities['resultado']:
                entities['resultado'][result_type].append(entity)
    
    def _identify_party_references(self, text):
        """
        Identify collective party references in the text.
        
        Args:
            text (str): The deliberation text
            
        Returns:
            list: Detected party references
        """
        party_references = []
        
        # Process text for collective party vote patterns
        for pattern in self.party_vote_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                party_ref = match.group(1).strip() + " " + match.group(0)[match.start(1):match.start(2)].strip()
                party = match.group(2).strip()
                vote_position = match.group(3).strip().lower()
                
                vote_type = "unknown"
                if any(term in vote_position for term in ["favor", "favorável"]):
                    vote_type = "favor"
                elif any(term in vote_position for term in ["contra", "desfavorável"]):
                    vote_type = "contra"
                elif any(term in vote_position for term in ["abst", "abstenção"]):
                    vote_type = "abstencao"
                
                # Get character spans
                start_pos = match.start()
                ref_end_pos = match.start(3)
                full_end_pos = match.end()
                
                party_references.append({
                    'text': match.group(0),
                    'ref_text': text[start_pos:ref_end_pos].strip(),
                    'partido': party,
                    'tipo': vote_type,
                    'start': start_pos,
                    'end': full_end_pos,
                    'ref_start': start_pos,
                    'ref_end': ref_end_pos,
                    'reference_type': 'collective'
                })
        
        # Also identify gendered party references
        for ref_type, patterns in self.party_reference_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    party_name = match.group(1).strip()
                    
                    party_references.append({
                        'text': match.group(0),
                        'ref_text': match.group(0),
                        'partido': party_name,
                        'tipo': None,  # No vote type specified in this reference
                        'start': match.start(),
                        'end': match.end(),
                        'ref_start': match.start(),
                        'ref_end': match.end(),
                        'reference_type': ref_type
                    })
        
        return party_references
    
    def _resolve_party_references(self, party_references, party_to_participants):
        """
        Map party references to individual participants using provided context.
        
        Args:
            party_references (list): List of detected party references
            party_to_participants (dict): Mapping from party names to participants
            
        Returns:
            list: Party references with resolved participants
        """
        resolved_references = []
        
        for ref in party_references:
            party_name = ref['partido']
            clean_party = remove_accents(party_name).lower()
            matched_party = None
            matched_participants = []
            
            # Try to find an exact match
            for known_party, participants in party_to_participants.items():
                clean_known_party = remove_accents(known_party).lower()
                
                # Check for exact or substring match
                if clean_party == clean_known_party or clean_party in clean_known_party or clean_known_party in clean_party:
                    matched_party = known_party
                    matched_participants = participants
                    break
            
            # If we have a match and it's a gendered reference, filter by gender
            if matched_participants:
                filtered_participants = matched_participants
                
                # Filter by gender if this is a gender-specific reference
                if 'reference_type' in ref:
                    ref_type = ref['reference_type']
                    if 'female' in ref_type:
                        filtered_participants = [
                            p for p in matched_participants 
                            if p.get('nome', '').strip().endswith('a') 
                            and not p.get('nome', '').strip().lower().endswith(('costa', 'rocha'))
                        ]
                    elif 'male' in ref_type:
                        filtered_participants = [
                            p for p in matched_participants 
                            if not p.get('nome', '').strip().endswith('a') 
                            or p.get('nome', '').strip().lower().endswith(('costa', 'rocha'))
                        ]
                
                # Add vote type to participants if available in the reference
                enriched_participants = []
                vote_type = ref.get('tipo')
                for p in filtered_participants:
                    participant_copy = p.copy()
                    # Only override participant's type if the reference specifies a vote type
                    if vote_type:
                        participant_copy['tipo'] = vote_type
                    enriched_participants.append(participant_copy)
                
                resolved_reference = ref.copy()
                resolved_reference.update({
                    'matched_party': matched_party,
                    'participants': enriched_participants
                })
                resolved_references.append(resolved_reference)
        
        return resolved_references


if __name__ == "__main__":
    # Example usage
    model = VoteIdentifierInference()
    
    example_text = """
    Na votação deste assunto, os vereadores do PS votaram a favor, enquanto os vereadores do PSD votaram contra.
    O vereador João Silva absteve-se.
    A proposta foi aprovada por maioria.
    """
    
    # Example context with party to participant mapping
    example_context = {
        'party_to_participants': {
            'PS': [
                {'id': '1', 'nome': 'Maria Santos', 'tipo': 'favor'},
                {'id': '2', 'nome': 'António Ferreira', 'tipo': 'favor'},
                {'id': '3', 'nome': 'Ana Costa', 'tipo': 'favor'}
            ],
            'PSD': [
                {'id': '4', 'nome': 'José Pereira', 'tipo': 'contra'},
                {'id': '5', 'nome': 'Carlos Rocha', 'tipo': 'contra'}
            ]
        }
    }
    
    # Run prediction
    results = model.predict(example_text, example_context)
    
    # Print results
    print("DETECTED ENTITIES:")
    print(f"Voters: {len(results['entities']['votante'])}")
    for voter in results['entities']['votante']:
        print(f"  - {voter['text']} (pos {voter['start']}:{voter['end']})")
    
    print("\nPARTY REFERENCES:")
    for ref in results['party_references']:
        print(f"  - {ref['ref_text']} ({ref['partido']}: {ref['tipo']})")
    
    print("\nRESOLVED REFERENCES:")
    for ref in results['resolved_references']:
        print(f"  - {ref['ref_text']} ({ref['partido']} -> {ref['matched_party']})")
        print(f"    Vote type: {ref['tipo']}")
        print("    Participants:")
        for p in ref['participants']:
            print(f"      * {p['nome']} ({p['tipo']})")