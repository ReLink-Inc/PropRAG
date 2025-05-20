"""
Beam search implementation for PropRAG's entity-based knowledge graph
"""

import os
import copy
import logging
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from collections import defaultdict
import itertools
import torch

from .utils.misc_utils import compute_mdhash_id
from .prompts.linking import get_query_instruction

logger = logging.getLogger(__name__)

class BeamSearchPathFinder:
    """
    Implements beam search for finding relevant proposition paths in PropRAG.
    
    This class works with PropRAG's graph (when use_propositions=True), where
    the graph contains entity nodes but no proposition nodes.
    
    The beam search algorithm:
    1. Finds high-scoring starting propositions based on query relevance
    2. Efficiently finds connected propositions through shared or synonymous entities
    3. Builds and scores paths of propositions based on relevance to the query
    4. Returns the best paths to improve retrieval quality
    """
    
    def __init__(self, prop_rag, beam_width: int = 4, max_path_length: int = 3, 
                 embedding_combination: str = "concatenate", sim_threshold: float = 0.75,
                 embedding_predictor = None, second_stage_filter_k: int = 0, debug=False,
                 subgraph=None):
        """
        Initialize the beam search path finder.
        
        Args:
            prop_rag: The PropRAG instance containing the knowledge graph
            beam_width: Number of top paths to keep at each step
            max_path_length: Maximum length of proposition paths to consider
            embedding_combination: Method to combine proposition embeddings:
                                  - "concatenate": Re-embed the concatenated text
                                  - "average": Use average of individual proposition embeddings
                                  - "weighted_average": Weight by position in chain
                                  - "max_pool": Use element-wise maximum values
                                  - "attention": Use attention mechanism based on query relevance
                                  - "predictor": Use a pre-trained model to predict the combined embedding
            sim_threshold: Threshold for considering edges as synonyms (if non-integer weight >= threshold)
            embedding_predictor: Trained model for predicting combined embeddings (used with embedding_combination="predictor")
            second_stage_filter_k: If > 0, applies a second stage filtering using the concatenate method 
                                  on the top K candidates. Set to 0 to disable second stage filtering.
            subgraph: Optional subgraph to use instead of the full knowledge graph
        """
        self.rag = prop_rag
        self.beam_width = beam_width
        self.max_path_length = max_path_length
        self.embedding_combination = embedding_combination
        self.sim_threshold = sim_threshold
        self.embedding_predictor = embedding_predictor
        self.second_stage_filter_k = second_stage_filter_k
        self.debug = debug
        self.subgraph = subgraph  # Store the subgraph if provided
        self.active_graph = subgraph if subgraph is not None else self.rag.graph  # Use subgraph if provided, otherwise use full graph
        self.node_name_to_vertex_idx = self.rag.node_name_to_vertex_idx
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # Set up device for embedding predictor
        self.device = None
        if self.embedding_predictor is not None:
            import torch
            self.device = next(self.embedding_predictor.parameters()).device
            logger.debug(f"Embedding predictor is using device: {self.device}")
        
        # Build entity-to-proposition mapping (inverse of proposition_to_entities_map)
        self.entity_to_propositions_map = defaultdict(list)
        if hasattr(self.rag, 'proposition_to_entities_map'):
            for prop_key, entities in self.rag.proposition_to_entities_map.items():
                for entity in entities:
                    entity_key = compute_mdhash_id(entity, prefix="entity-")
                    self.entity_to_propositions_map[entity_key].append(prop_key)
            logger.debug(f"Built entity-to-proposition map with {len(self.entity_to_propositions_map)} entities")
        else:
            logger.warning("PropRAG instance doesn't have proposition_to_entities_map")
        
        self.entity_orig_idx_to_sub_idx_map = None
        # Cache maps to avoid redundant computations
        self.clear_caches()

    def set_node_name_to_vertex_idx(self, node_name_to_vertex_idx):
        self.node_name_to_vertex_idx = node_name_to_vertex_idx
        
    def clear_caches(self):
        """Clear all caches to refresh between iterations"""
        self.proposition_text_cache = {}
        self.proposition_embedding_cache = {}
        self.entity_text_cache = {}
        self.synonymous_entities_cache = {}
        self.connected_propositions_cache = {}
        
    def set_subgraph(self, subgraph):
        """Set a new subgraph to use for beam search"""
        self.subgraph = subgraph
        self.active_graph = subgraph if subgraph is not None else self.rag.graph
        self.clear_caches()  # Clear caches when changing the active graph

        self.predictor_model_path = "embedding_prediction_models/embedding_predictor_advanced_optimized_combined_4emb.pt"
        if embedding_combination == "predictor" and self.embedding_predictor is None:
            import sys
            
            # We need to import EmbeddingPredictionModel from test_embedding_prediction.py
            try:
                sys.path.append(os.getcwd())
                from test_embedding_prediction import EmbeddingPredictionModel
            except ImportError as e:
                logger.error(f"Error importing EmbeddingPredictionModel: {e}")
                logger.error("Please make sure test_embedding_prediction.py is in the current directory")
                logger.error("Falling back to 'average' method")
                self.embedding_combination = "average"
            
            # Check if model file exists
            if not os.path.exists(self.predictor_model_path):
                logger.error(f"Embedding prediction model not found at {self.predictor_model_path}")
                logger.error("Please make sure the model file exists or use a different embedding combination method")
                logger.error("Falling back to 'average' method")
                self.embedding_combination = "average"
            else:
                logger.debug(f"Loading embedding prediction model from {self.predictor_model_path}")
                import torch
                # Set device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logger.debug(f"Embedding predictor is using device: {self.device}")
                
                # Load model
                try:
                    checkpoint = torch.load(self.predictor_model_path, map_location=self.device)
                    
                    # Extract model parameters
                    embedding_dim = checkpoint.get('embedding_dim', self.rag.embedding_model.embedding_dim)
                    model_type = checkpoint.get('model_type', 'advanced_optimized')
                    
                    # Initialize model
                    self.embedding_predictor = EmbeddingPredictionModel(embedding_dim, model_type=model_type)
                    self.embedding_predictor.load_state_dict(checkpoint['model_state_dict'])
                    self.embedding_predictor.to(self.device)
                    self.embedding_predictor.eval()
                    
                    # Log model details
                    logger.debug(f"Loaded embedding prediction model of type {model_type}")
                    logger.debug(f"Embedding dimension: {embedding_dim}")
                    logger.debug(f"Best validation similarity: {checkpoint.get('val_sim', 'N/A')}")
                    
                    # Check if the model was trained for a specific number of embeddings
                    num_embeddings = checkpoint.get('num_embeddings', 'variable')
                    logger.debug(f"Model was trained for {num_embeddings} embeddings")
                    
                except Exception as e:
                    logger.error(f"Error loading embedding prediction model: {e}")
                    logger.error("Falling back to 'average' method")
                    self.embedding_combination = "average"
                    self.embedding_predictor = None
            
    def get_proposition_text(self, prop_key: str) -> str:
        """Get the text for a proposition by its key."""
        
        prop_row = self.rag.proposition_embedding_store.get_row(prop_key)
        if prop_row:
            prop_text = prop_row.get("content", "")
            return prop_text
        
        return ""
    
    def get_entity_text(self, entity_key: str) -> str:
        """Get the text for an entity by its key."""
        entity_row = self.rag.entity_embedding_store.get_row(entity_key)
        if entity_row:
            entity_text = entity_row.get("content", "")
            return entity_text
        
        return ""
    
    def get_orig_proposition_embedding(self, prop_key: str) -> np.ndarray:
        """Get the embedding for a proposition by its key."""
        return self.rag.proposition_embedding_store.get_embedding(prop_key)

    def get_proposition_embedding(self, prop_key: str) -> np.ndarray:
        """Get the embedding for a proposition by its key."""
        return self.rag.prop_key_to_propositions[prop_key]
        
        # return None
    def get_proposition_embeddings(self, prop_keys: List[str]) -> np.ndarray:
        """Get the embedding for a proposition by its key."""
        # return self.rag.proposition_embedding_store.get_embeddings(prop_keys)
        return np.array([self.rag.prop_key_to_propositions[prop_key] for prop_key in prop_keys])
    
    def find_entities_in_proposition(self, prop_key: str) -> List[str]:
        """Find the entities in a proposition."""
        entities = []
        if hasattr(self.rag, 'proposition_to_entities_map') and prop_key in self.rag.proposition_to_entities_map:
            entity_texts = self.rag.proposition_to_entities_map[prop_key]
            for entity_text in entity_texts:
                entity_key = compute_mdhash_id(entity_text, prefix="entity-")
                entities.append(entity_key)
        else:
            logger.warning(f"Proposition {prop_key} not found in proposition_to_entities_map")
            return []
        return entities
    
    def find_synonymous_entities(self, entity_key: str) -> List[Tuple[str, float]]:
        """
        Find entities connected via synonymy edges to the given entity.
        
        Args:
            entity_key: The entity key to find synonyms for
            
        Returns:
            List of (synonymous_entity_key, similarity_score) tuples
        """
        # Check cache first
        if entity_key in self.synonymous_entities_cache:
            return self.synonymous_entities_cache[entity_key]
        
        synonyms = []
            
        entity_idx = self.node_name_to_vertex_idx[entity_key]
        
        neighbor_set = set()
        # Get all neighbors of this entity
        try:
            # Use the active graph (subgraph or full graph)
            neighbors = self.active_graph.neighbors(entity_idx, mode="all")
            
            # For each neighbor, check if it's an entity and if the edge is a synonymy edge
            for neighbor_idx in neighbors:
                neighbor_name = self.active_graph.vs[neighbor_idx]["name"]
                if neighbor_name in neighbor_set:
                    continue
                neighbor_set.add(neighbor_name)
                
                # Skip if not an entity node
                if not neighbor_name.startswith("entity-"):
                    continue
                
                # Get edge between entities
                edge_id = self.active_graph.get_eid(entity_idx, neighbor_idx, error=False)
                if edge_id != -1:
                    edge_weight = self.active_graph.es[edge_id]["weight"]
                    
                    # Check if it's a synonymy edge (non-integer weight above threshold)
                    if (isinstance(edge_weight, float) and 
                        not edge_weight.is_integer() and 
                        edge_weight >= self.sim_threshold):
                        synonyms.append((neighbor_name, float(edge_weight)))
        
        except Exception as e:
            logger.warning(f"Error finding synonymous entities: {e}")
        
        # Cache the result
        self.synonymous_entities_cache[entity_key] = synonyms
        return synonyms
    
    def find_connected_propositions(self, prop_key: str, prop_set: List[str] = None) -> List[Tuple[str, List[Dict]]]:
        """
        Find propositions connected to the given proposition efficiently.
        
        A proposition is connected if it shares an entity with the given proposition
        or has an entity that is synonymous with an entity in the given proposition.
        
        Args:
            prop_key: The proposition key
            
        Returns:
            List of (connected_prop_key, connection_details) tuples
        """
        # if prop_set is not None:
        #     return [(prop, []) for prop in prop_set], set(prop_set)
        # Check cache first
        if prop_key in self.connected_propositions_cache:
            return self.connected_propositions_cache[prop_key]
        
        connected_props = []
        
        # Get entities in this proposition
        prop_entities = self.find_entities_in_proposition(prop_key)
        if not prop_entities:
            self.connected_propositions_cache[prop_key] = (connected_props, set())
            return self.connected_propositions_cache[prop_key]
        
        # Track which propositions we've already processed
        processed_props = set()
        
        # Find connections through exact entity matches and synonyms
        for entity_key in prop_entities:
            if entity_key in self.connected_propositions_cache:
                propositions_connected_by_entity = self.connected_propositions_cache[entity_key]
                connected_props.extend([(other_prop_key, _) for other_prop_key, _ in propositions_connected_by_entity if (
                    other_prop_key != prop_key and other_prop_key not in processed_props and (prop_set is None or other_prop_key in prop_set))])
                processed_props.update([p for p, _ in propositions_connected_by_entity])
                continue
            entity_processed_props = set()
            connected_props_by_entity = []
            # First, get propositions sharing this exact entity
            exact_props = self.entity_to_propositions_map.get(entity_key, [])
            for other_prop_key in exact_props:
                # Skip self and already processed
                if other_prop_key in entity_processed_props:
                    continue
                
                # This is a connection via exact entity match
                connection = {
                    "type": "exact",
                    "entity1": entity_key,
                    "entity2": entity_key,
                    "similarity": 1.0
                }
                connected_props_by_entity.append((other_prop_key, [connection]))
                entity_processed_props.add(other_prop_key)
            
            # Next, find synonymous entities and their propositions
            synonyms = self.find_synonymous_entities(entity_key)
            for synonym_key, similarity in synonyms:
                # Get propositions containing this synonym entity
                synonym_props = self.entity_to_propositions_map.get(synonym_key, [])
                for other_prop_key in synonym_props:
                    # Skip self and already processed
                    if other_prop_key in entity_processed_props:
                        continue
                    
                    # This is a connection via synonymous entities
                    connection = {
                        "type": "synonym",
                        "entity1": entity_key,
                        "entity2": synonym_key,
                        "similarity": similarity
                    }
                    
                    connected_props_by_entity.append((other_prop_key, [connection]))
                    entity_processed_props.add(other_prop_key)
            # if entity_key == 'entity-American':
            #     print("entity_key: ", entity_key)
            #     print("prop_key: ", prop_key)
            #     for conn in connected_props_by_entity:
            #         connection = conn[1][0]
            #         print("conn_type: ", connection["type"], "from entity: ", connection["entity1"], "to_entity: ", connection["entity2"], "conn_similarity: ", connection["similarity"])
            #         print("other_prop_key: ", conn[0])
            #     print("--------------------------------")
            self.connected_propositions_cache[entity_key] = connected_props_by_entity
            connected_props.extend([(other_prop_key, _) for other_prop_key, _ in connected_props_by_entity if (
                other_prop_key != prop_key and other_prop_key not in processed_props and (prop_set is None or other_prop_key in prop_set))])
            processed_props.update(entity_processed_props)
        # Cache the result
        self.connected_propositions_cache[prop_key] = (connected_props, processed_props)
        return connected_props, processed_props
        
    def batch_score_propositions(self, prop_keys: List[str], query_embedding: np.ndarray) -> List[float]:
        """
        Score multiple propositions' relevance to the query in a single batch operation.
        
        Args:
            prop_keys: List of proposition keys to score
            query_embedding: Query embedding
            
        Returns:
            List of relevance scores in the same order as prop_keys
        """
        # Get embeddings for all propositions
        embeddings_matrix = self.get_proposition_embeddings(prop_keys)
        
        # Calculate all scores in one dot product operation
        scores_array = np.dot(embeddings_matrix, query_embedding.T).squeeze()
        
        # Convert to list of floats
        if scores_array.ndim == 0:
            scores_array = np.array([float(scores_array)])
        else:
            scores_array = scores_array.astype(float)
        
        return scores_array
    
        
    def batch_score_paths(self, paths: List[Dict], query_embedding: np.ndarray) -> List[float]:
        """
        Score multiple paths in batch based on their relevance to the query.
        
        Args:
            paths: List of path dictionaries
            query_embedding: Query embedding
            
        Returns:
            List of relevance scores in the same order as paths
        """
        if not paths:
            return []
            
        # Extract proposition keys from each path
        paths_prop_keys = [path["propositions"] for path in paths]
        
        # Check if all paths have the same length (required for efficient batch processing)
        path_lengths = [len(props) for props in paths_prop_keys]
        if len(set(path_lengths)) > 1:
            raise ValueError("All paths must have the same length for batch processing")
        
        path_length = path_lengths[0]  # Length is same for all paths
        
        # Get combined embeddings using the appropriate method
        if self.embedding_combination == "predictor" and self.embedding_predictor is not None:
            # First, compute simple average embeddings for all paths
            # Get all embeddings at once to avoid loops
            all_embeddings = []
            for path_props in paths_prop_keys:
                path_embeddings = list(self.get_proposition_embeddings(path_props))
                all_embeddings.append(path_embeddings)
                
            # Convert to numpy array for batch operations
            # Shape: [num_paths, path_length, embedding_dim]
            embeddings_batch = np.array(all_embeddings)
            
            # Simple averaging for all paths at once: [num_paths, embedding_dim]
            combined_embs = np.mean(embeddings_batch, axis=1)
            norms = np.linalg.norm(combined_embs, axis=1, keepdims=True)
            normalized_embs = combined_embs / norms
            
            # Calculate scores for all paths at once
            query_vec = query_embedding.squeeze()
            initial_scores = np.matmul(normalized_embs, query_vec).tolist()
            
            # Find indices of top 100 scores (or fewer if there are less than 100 paths)
            top_k = min(100, len(initial_scores))
            top_indices = np.argsort(initial_scores)[-top_k:][::-1]  # Get indices of top scores in descending order
            
            # Initialize final scores with zeros
            final_scores = [0.0] * len(paths)
            
            # Only process the top 100 paths with the predictor model
            if top_indices.size > 0:
                import torch
                
                # Setup for batch processing (process multiple paths at once)
                batch_size = 64  # Adjust based on GPU memory
                
                # Get the paths to process with the predictor
                paths_to_process = [paths_prop_keys[i] for i in top_indices]
                
                # Process in batches
                predictor_scores = []
                for i in tqdm(range(0, len(paths_to_process), batch_size), desc="Scoring paths with predictor", disable=not self.debug):
                    batch_paths = paths_to_process[i:i+batch_size]
                    current_batch_size = len(batch_paths)
                    
                    # Extract all proposition embeddings for this batch
                    # Shape will be [batch_size, path_length, embedding_dim]
                    path_embs_list = []
                    for j in range(path_length):
                        # Get the j-th embedding from each path
                        prop_keys_at_pos_j = [batch_paths[path_idx][j] for path_idx in range(current_batch_size)]
                        embs_at_pos_j = self.get_proposition_embeddings(prop_keys_at_pos_j)
                        
                        # Convert to batch tensor [batch_size, embedding_dim]
                        path_embs_list.append(torch.tensor(embs_at_pos_j, dtype=torch.float32).to(self.device))
                    
                    # Process entire batch at once with predictor model
                    with torch.no_grad():
                        self.embedding_predictor.eval()
                        
                        # Forward pass through the predictor model
                        # The model expects multiple embeddings, one for each position in the path
                        predicted_embs = self.embedding_predictor(*path_embs_list)
                        
                        # Convert query to tensor for batch dot product
                        query_tensor = torch.tensor(query_embedding.squeeze(), 
                                                   dtype=torch.float32).to(self.device)
                        
                        # Calculate scores for entire batch at once
                        # Using batch dot product between predicted embeddings and query
                        batch_scores = torch.matmul(predicted_embs, query_tensor).cpu().numpy()
                        predictor_scores.extend(batch_scores)
                
                # Assign predictor scores to the top paths
                for idx, score in zip(top_indices, predictor_scores):
                    final_scores[idx] = score
            
            return final_scores
        else:
            # For other embedding combination methods, use numpy batch operations
            # Gather all embeddings at once to avoid loops
            all_embeddings = []
            for path_props in paths_prop_keys:
                path_embeddings = list(self.get_proposition_embeddings(path_props))
                all_embeddings.append(path_embeddings)
                
            # Convert to numpy array for batch operations
            # Shape: [num_paths, path_length, embedding_dim]
            embeddings_batch = np.array(all_embeddings)
            embedding_dim = embeddings_batch.shape[2]
            num_paths = embeddings_batch.shape[0]
            
            # Apply batch operations for each combination method
            if self.embedding_combination == "concatenate":
                # For concatenate, we need to re-encode the concatenated text for each path
                combined_embs = np.zeros((num_paths, embedding_dim))
                
                # Process paths in smaller batches to avoid memory issues

                batch_paths = paths_prop_keys
                batch_len = len(batch_paths)
                
                # Get all proposition texts for this batch
                batch_prop_texts = []
                for path_props in batch_paths:
                    path_texts = [self.get_proposition_text(prop_key) for prop_key in path_props]
                    concatenated_text = " ".join(path_texts)
                    batch_prop_texts.append(concatenated_text)
                
                # Batch encode all concatenated texts at once
                combined_embs = self.rag.embedding_model.batch_encode(
                    batch_prop_texts, norm=True, disable_tqdm=not self.debug, batch_size=8
                )
                
            elif self.embedding_combination == "average":
                # Simple averaging for all paths at once: [num_paths, embedding_dim]
                combined_embs = np.mean(embeddings_batch, axis=1)
                
            elif self.embedding_combination == "weighted_average":
                # Create weights: linear from 0.5 to 1.0, same for all paths
                weights = np.linspace(0.5, 1.0, path_length)
                weights = weights / weights.sum()  # Normalize
                
                # Reshape weights to [1, path_length, 1] for broadcasting
                weights_reshaped = weights.reshape(1, path_length, 1)
                
                # Weighted sum across path_length dimension for all paths at once
                combined_embs = np.sum(embeddings_batch * weights_reshaped, axis=1)
                
            elif self.embedding_combination == "max_pool":
                # Find max values for each dimension across path elements
                abs_embeddings = np.abs(embeddings_batch)
                max_indices = np.argmax(abs_embeddings, axis=1)
                
                # Create index arrays for batch extraction
                batch_indices = np.arange(num_paths).repeat(embedding_dim)
                dim_indices = np.tile(np.arange(embedding_dim), num_paths)
                flat_max_indices = max_indices.reshape(-1)
                
                # Extract values from original embeddings using calculated indices
                combined_embs = embeddings_batch[batch_indices, flat_max_indices, dim_indices].reshape(num_paths, embedding_dim)
                
            elif self.embedding_combination == "attention":
                # Batch implementation of attention mechanism
                query_vec = query_embedding.squeeze()
                
                # Calculate attention scores for all paths at once
                # Reshape for broadcasting: [num_paths, path_length, embedding_dim] @ [embedding_dim]
                query_similarities = np.matmul(embeddings_batch, query_vec)
                
                # Apply softmax to each path's attention scores
                exp_similarities = np.exp(query_similarities)
                attention_weights = exp_similarities / np.sum(exp_similarities, axis=1, keepdims=True)
                
                # Apply attention weights to embeddings
                # Reshape attention weights to [num_paths, path_length, 1]
                attention_weights_reshaped = attention_weights.reshape(num_paths, path_length, 1)
                combined_embs = np.sum(embeddings_batch * attention_weights_reshaped, axis=1)
                
            else:
                # Default to average
                combined_embs = np.mean(embeddings_batch, axis=1)
            
            # Normalize all embeddings at once
            norms = np.linalg.norm(combined_embs, axis=1, keepdims=True)
            normalized_embs = combined_embs / norms
            
            # Calculate scores for all paths at once
            query_vec = query_embedding.squeeze()
            scores = np.matmul(normalized_embs, query_vec)
            
            return scores.tolist()

    def find_paths(self, query: str, prop_set: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find and score paths of connected propositions that are relevant to the query.
        
        If use_second_stage_filter is True, this will:
        1. Score paths with the configured embedding_combination method
        2. Take the top second_stage_filter_k candidates 
        3. Re-score these candidates using the concatenate method
        4. Return the top beam_width candidates from this second stage
        
        Args:
            query: The query string
            
        Returns:
            List of path dictionaries sorted by score. If second stage filtering 
            was applied, paths will also include a 'concatenate_score' field.
        """
        logger.debug(f"Starting beam search for query: {query}")
        # start_time = time.time()
        
        # Get query embedding for scoring
        query_embedding = self.rag.embedding_model.batch_encode(
            query, instruction=get_query_instruction('query_to_passage'), norm=True, disable_tqdm=not self.debug
        )
        
        # Score all propositions for initial candidates using batch processing
        logger.debug("Scoring propositions for initial candidates...")
        if prop_set is None:
            all_prop_keys = self.rag.proposition_embedding_store.get_all_ids()
            all_scores = np.dot(self.rag.all_proposition_embeddings, query_embedding.T).squeeze()
        else:
            all_prop_keys = prop_set
            all_scores = np.dot(self.get_proposition_embeddings(all_prop_keys), query_embedding.T).squeeze()
        # Process in batches to avoid memory issues
        # batch_size = len(all_prop_keys)
        # print(all_scores.shape)
        # dot_end_time = time.time()
        
        top_k = min(200, len(all_scores))
        top_indices = np.argpartition(all_scores, -top_k)[-top_k:]
        # Sort just these top indices by their scores
        top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        
        # Get the corresponding keys and scores
        top_prop_keys = [all_prop_keys[idx] for idx in top_indices]
        top_prop_scores = all_scores[top_indices]
        
        # Create the final list of (key, score) pairs
        top_props = list(zip(top_prop_keys, top_prop_scores))
        # sort_end_time = time.time()
        logger.debug(f"Selected {len(top_props)} initial propositions")
        
        # Initialize beam with single-proposition paths
        beam = []
        for prop_key, score in top_props:
            # Path format: propositions, connections, score
            path = {
                "propositions": [prop_key],
                "connections": [],
                "score": score
            }
            beam.append(path)
        
        # Select top initial paths
        beam.sort(key=lambda x: x["score"], reverse=True)

        keep_initial_props = 3
        initial_prop_set = set(path["propositions"][0] for path in beam[:keep_initial_props])
        initial_props = [(path["propositions"][0], []) for path in beam[:keep_initial_props]]

        beam = beam[:self.beam_width]
        initial_paths = copy.deepcopy(beam)
        
        # Track all paths for final selection
        all_paths = []
        all_path_sets = set()  # To prevent duplicate paths with same propositions
        
        # Add initial candidates to all_paths
        for path in beam:
            path_set = frozenset(path["propositions"])
            if path_set not in all_path_sets:
                all_path_sets.add(path_set)
                all_paths.append(path.copy())
        
        # Display top initial propositions
        logger.debug(f"Top {min(5, len(beam))} initial propositions:")
        for i, path in enumerate(beam[:5]):
            prop_text = self.get_proposition_text(path["propositions"][0])
            logger.debug(f"  {i+1}. Score: {path['score']:.4f} - {prop_text}")
        
        other_candidates = []
        # Build paths incrementally
        for depth in range(2, self.max_path_length + 1):
            logger.debug(f"Beam search depth {depth}/{self.max_path_length}")
            
            # Track new candidates for this depth
            new_candidates = []
            
            # For each path in the current beam
            # Group paths by last proposition to batch process
            paths_by_last_prop = {}
            
            for path_idx, path in enumerate(beam):
                last_prop = path["propositions"][-1]
                if last_prop not in paths_by_last_prop:
                    paths_by_last_prop[last_prop] = []
                paths_by_last_prop[last_prop].append((path_idx, path))
            
            # Process each group of paths with the same last proposition
            for last_prop, paths_with_prop in paths_by_last_prop.items():
                # Find connected propositions efficiently (this is a bottleneck, done once per group)
                connected_props, connected_props_set = self.find_connected_propositions(last_prop, prop_set=prop_set)

                # Explore non-chain reasonings (parallel reasoning)
                for prop in initial_props:
                    if last_prop != prop[0] and prop[0] not in connected_props_set:
                        connected_props.append(prop)
                
                # Allow all unconnected props
                # connected_props = [(prop, []) for prop in prop_set]
                
                # Create candidate paths for all paths in this group
                for path_idx, path in paths_with_prop:
                    # For each connected proposition
                    for next_prop, connections in connected_props:
                        # Skip if already in path
                        if next_prop in path["propositions"]:
                            continue
                        
                        # Create new path by extending current path
                        new_path = {
                            "propositions": path["propositions"] + [next_prop],
                            "connections": path["connections"] + [
                                {
                                    "from_prop": last_prop,
                                    "to_prop": next_prop,
                                    "entity_connections": connections,
                                }
                            ],
                            # "coherence": 1.0 if next_prop in initial_prop_set else np.dot(self.get_proposition_embedding(last_prop), self.get_proposition_embedding(next_prop))
                        }
                        # if next_prop in initial_prop_set:
                        #     print("next_prop in initial_prop_set: ", next_prop, "coherence: ", new_path["coherence"])
                        
                        # Create a unique path identifier to prevent duplicates
                        path_set = frozenset(new_path["propositions"])
                        if path_set in all_path_sets:
                            continue
                        
                        # Add to new candidates (will be scored in batch later)
                        new_path["score"] = 0.0  # Placeholder, will be updated in batch
                        new_candidates.append(new_path)
                        
                        # Add to seen paths
                        all_path_sets.add(path_set)
            
            # If no new candidates found, break
            if not new_candidates:
                logger.debug(f"No more valid paths found at depth {depth}")
                break
            
            # Score all candidates in batches (all have same length at this depth)
            batch_size = len(new_candidates)
            for i in tqdm(range(0, len(new_candidates), batch_size), desc="Scoring paths", disable=not self.debug):
                batch_paths = new_candidates[i:i+batch_size]
                batch_scores = self.batch_score_paths(batch_paths, query_embedding)
                
                # Update scores in place
                for j, score in enumerate(batch_scores):
                    new_candidates[i + j]["score"] = float(score)
            
            # Sort and select top candidates
            new_candidates.sort(key=lambda x: x["score"], reverse=True)
            # print("new_candidates length: ", len(new_candidates))
            # Apply second stage filtering if enabled
            if self.second_stage_filter_k > 0 and len(new_candidates) > self.beam_width:
                # Take top K candidates from first stage for second stage filtering
                first_stage_candidates = new_candidates[:min(len(new_candidates), self.second_stage_filter_k)]
                logger.debug(f"Applying second stage filtering with concatenate method on top {len(first_stage_candidates)} candidates")
                
                # Save original embedding combination method
                saved_embedding_combination = self.embedding_combination
                
                # Force concatenate method for second stage
                self.embedding_combination = "concatenate"
                
                # Score candidates using concatenate method
                batch_paths = first_stage_candidates
                batch_scores = self.batch_score_paths(batch_paths, query_embedding)

                # print("candidate scores: ", [(candidate["propositions"], candidate["connections"][-1]["entity_connections"], score) for candidate, score in zip(first_stage_candidates, batch_scores)])
                # Store scores under a different key to preserve original scores
                for j, score in enumerate(batch_scores):
                    first_stage_candidates[j]["concatenate_score"] = float(score)
                
                # Restore original embedding combination method
                self.embedding_combination = saved_embedding_combination
                
                # Sort by concatenate scores
                first_stage_candidates.sort(key=lambda x: x.get("concatenate_score", -1.1), reverse=True)
                new_candidates = first_stage_candidates[:self.second_stage_filter_k]
                # Select top paths based on concatenate scores
                for i, path in enumerate(new_candidates):
                    path['score'] = path['concatenate_score']
                
                # Log the second stage filtering results
                logger.debug(f"Second stage filtering complete, selected top {len(beam)} paths")
                # other_candidates.append(sorted(new_candidates, key=lambda x: x['score'], reverse=True))
            
            scores = np.array([path["score"] for path in new_candidates])
            # coherences = np.array([path["coherence"] for path in new_candidates])
            
            # scores[coherences > 0.05] += 1 # best threshold for pure proposition embeddings
            # TODO: revert the coherence changes

            sorted_indices = np.argsort(-scores)
            new_candidates = [new_candidates[i] for i in sorted_indices]
            
            # # Get the ranks (argsort of argsort gives the ranks)
            # # We negate the values because we want descending order (highest first)
            # score_ranks = np.argsort(np.argsort(-scores))
            # coherence_ranks = np.argsort(np.argsort(-coherences))
            
            # # Compute combined ranks
            # combined_ranks = score_ranks + coherence_ranks
            # # combined_ranks = score_ranks 
            
            # # Sort new_candidates based on combined ranks
            # sorted_indices = np.argsort(combined_ranks)
            # new_candidates = [new_candidates[i] for i in sorted_indices]
            
            beam = new_candidates[:self.beam_width]
            # Add new paths to all_paths
            for path in new_candidates[:self.beam_width]:
                all_paths.append(path.copy())
            
            if depth < self.max_path_length:
                other_candidates.append(sorted(all_paths, key=lambda x: x['score'], reverse=True)[:self.beam_width])

            # Log progress
            logger.debug(f"Found {len(new_candidates)} new paths, keeping top {len(beam)}")
            # self.debug = True
            if beam and self.debug:
                for i, path in enumerate(beam[:5]):
                    prop_text = " ".join([self.get_proposition_text(prop) for prop in path["propositions"]])
                    logger.info(f"  {i+1}. Score: {path['score']:.4f} - {path['coherence']:.4f} - {prop_text}")
            # self.debug = False
        
        # Post-process paths: add proposition texts and entity texts
        processed_paths = []
        # print("all_paths length: ", len(all_paths))
        for path in all_paths:
            # print(path)
            # Get proposition texts
            prop_texts = []
            for prop_key in path["propositions"]:
                prop_text = self.get_proposition_text(prop_key)
                prop_texts.append(prop_text)
            
            # Get entity texts for connections
            connections_info = []
            for conn in path["connections"]:
                conn_details = []
                for connection in conn["entity_connections"]:
                    # Get entity texts
                    entity1_key = connection["entity1"]
                    entity2_key = connection["entity2"]
                    entity1_text = self.get_entity_text(entity1_key)
                    entity2_text = self.get_entity_text(entity2_key)
                    
                    conn_details.append({
                        "entity1": entity1_text,
                        "entity2": entity2_text,
                        "type": connection["type"],
                        "similarity": connection.get("similarity", 1.0)
                    })
                
                from_prop_text = self.get_proposition_text(conn["from_prop"])
                to_prop_text = self.get_proposition_text(conn["to_prop"])
                
                connections_info.append({
                    "from_proposition": from_prop_text,
                    "to_proposition": to_prop_text,
                    "entity_connections": conn_details
                })
            
            processed_path = {
                "proposition_keys": path["propositions"],
                "proposition_texts": prop_texts,
                "connections": connections_info,
                "score": path.get("score", 0.0),
                "length": len(path["propositions"])
            }
            
            # Include concatenate_score if available (from second stage filtering)
            if "concatenate_score" in path:
                processed_path["concatenate_score"] = path["concatenate_score"]
                
            processed_paths.append(processed_path)
        
        # print("processed paths: ", processed_paths)
        # Sort all paths by score
        processed_paths.sort(key=lambda x: x["score"], reverse=True)
        # print("processed paths length: ", len(processed_paths))
        
        # Log final results
        logger.debug(f"Found {len(processed_paths)} unique paths across all depths")
        logger.debug(f"Best path score: {processed_paths[0]['score'] if processed_paths else 0.0:.4f}")
        # post_process_end_time = time.time()
        # print(f"Time taken for dot product: {dot_end_time - start_time:.4f} seconds")
        # print(f"Time taken for sort: {sort_end_time - dot_end_time:.4f} seconds")
        # print(f"Time taken for post-process: {post_process_end_time - sort_end_time:.4f} seconds")
        # Return top paths
        return processed_paths, [[{"proposition_keys": path['propositions'], "score": path['score']} for path in candidates] for candidates in other_candidates], [{"proposition_keys": path['propositions'], "score": path['score']} for path in initial_paths]

    def get_propositions_from_paths(self, paths: List[Dict[str, Any]]) -> List[str]:
        """
        Get the most important propositions from the top paths.
        
        Args:
            paths: List of path dictionaries from find_paths
            top_k: Maximum number of propositions to return
            
        Returns:
            List of proposition keys
        """
        # Count propositions across all paths, weighted by path score
        prop_scores = defaultdict(list)
        
        for path in paths:
            path_score = path["score"]
            for prop_key in path["proposition_keys"]:
                prop_scores[prop_key].append(path_score)
        
        # Sort propositions by score
        sorted_props = sorted(prop_scores.items(), key=lambda x: np.sum(x[1]), reverse=True)
        
        # Return top-k proposition keys
        return [(prop, score) for prop, score in sorted_props]
    
    def get_entities_from_paths(self, paths: List[Dict[str, Any]]) -> List[str]:
        """
        Get the most important entities from the top paths.
        
        Args:
            paths: List of path dictionaries from find_paths
            
        Returns:
            List of entity keys
        """
        # Count entities across all paths, weighted by path score
        entity_scores = defaultdict(list)
        
        for path in paths:
            path_score = path["score"]

            entity_conns = defaultdict(list)
            if 'connections' in path:
                connections = path['connections']
                for path_conn_info in connections:
                    for conn_detail in path_conn_info['entity_connections']:
                        if conn_detail['entity1'] != conn_detail['entity2']:           
                            # Boost the score of the connected entity instead of the connecting entity.
                            entity_conns[conn_detail['entity1']].append(compute_mdhash_id(conn_detail['entity2'], prefix="entity-"))
            for _, conn_entity_keys in entity_conns.items():
                for conn_entity_key in conn_entity_keys:
                    entity_scores[conn_entity_key].append(path_score)
            # Extract entity keys from all propositions in the path
            for prop_key in path["proposition_keys"]:
                entity_keys = self.find_entities_in_proposition(prop_key)
                for entity_key in entity_keys:
                    entity_scores[entity_key].append(path_score)
        
        # Sort entities by score
        sorted_entities = sorted(entity_scores.items(), key=lambda x: np.sum(x[1]), reverse=True)
        
        # Return top-k entity keys
        return [(entity, scores) for entity, scores in sorted_entities]

    def get_connected_entities_from_paths(self, paths: List[Dict[str, Any]]) -> List[str]:
        """
        Get the most important connected entities from the top paths.
        
        Args:
            paths: List of path dictionaries from find_paths
            top_k: Maximum number of connected entities to return

        Returns:
            List of connected entity keys
        """
        # Count connected entities across all paths, weighted by path score
        connected_entity_scores = defaultdict(float)
        
        for path in paths:
            path_score = path["score"]
            for conn in path["connections"]:
                for entity_connection in conn["entity_connections"]:
                    entity1_key = entity_connection["entity1"]
                    entity2_key = entity_connection["entity2"]
                    connected_entity_scores[entity1_key] += path_score
                    if entity1_key != entity2_key:
                        connected_entity_scores[entity2_key] += path_score
        
        # Sort connected entities by score
        sorted_connected_entities = sorted(connected_entity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k connected entity keys
        return [(compute_mdhash_id(entity, prefix="entity-"), score) for entity, score in sorted_connected_entities]
        