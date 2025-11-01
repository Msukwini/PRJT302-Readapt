"""
Readapt Adaptive Learning System - Data Collection & Processing Pipeline
Implements unsupervised learning methodology from conference paper
"""

import os
import json
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import spacy
from collections import defaultdict
import re


def is_folder_empty_pathlib(folder_path_str):
    """Checks if a folder is empty using pathlib."""
    folder_path = Path(folder_path_str)
    try:
        # Check if the path exists and is a directory first (optional but good practice)
        if not folder_path.is_dir():
            print(f"Error: Path is not an existing directory at '{folder_path_str}'")
            return False
        
        # Use not any() on the iterator returned by iterdir()
        return not any(folder_path.iterdir())
    except OSError as e:
        print(f"An OS error occurred: {e}")
        return False


# ===========================
# 1. KAGGLE DATA COLLECTION
# ===========================

class KaggleDataCollector:
    """Downloads and organizes datasets from Kaggle API"""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'race': 'ankitdhiman7/race-dataset',
            'squad': 'stanfordu/stanford-question-answering-dataset',
            'cefr_texts': 'amontgomerie/cefr-levelled-english-texts',
            'mmlu': 'peiyuanliu2001/mmlu-dataset',
            'iot_learning': 'ziya07/language-learning-analysis-with-iot',
            'en_cn_learning': 'datasetengineer/english-chinese-learning-dataset',
            'fasttext_news': 'facebook/fasttext-wikinews'
        }
        
    def setup_kaggle_api(self):
        """Initialize Kaggle API - requires kaggle.json in ~/.kaggle/"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle API authenticated successfully")
            return api
        except Exception as e:
            print(f"❌ Kaggle API setup failed: {e}")
            print("Please ensure kaggle.json is in ~/.kaggle/ directory")
            return None
    
    def download_datasets(self):
        """Download all configured datasets"""
        api = self.setup_kaggle_api()
        if not api:
            return False
        
        print(f"\n📥 Starting dataset downloads to: {self.base_dir.absolute()}")
        print("=" * 60)
        
        for name, dataset_path in self.datasets.items():
            download_dir = self.base_dir / name
            download_dir.mkdir(exist_ok=True)
            
            try:
                print(f"\n⏳ Downloading {name}: {dataset_path}")
                api.dataset_download_files(
                    dataset_path,
                    path=download_dir,
                    unzip=True,
                    quiet=False
                )
                print(f"✅ {name} downloaded successfully")
                
            except Exception as e:
                print(f"⚠️  Failed to download {name}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("📊 Download Summary:")
        self._print_dataset_summary()
        return True
    
    def _print_dataset_summary(self):
        """Print summary of downloaded datasets"""
        for name in self.datasets.keys():
            dataset_dir = self.base_dir / name
            if dataset_dir.exists():
                files = list(dataset_dir.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  • {name}: {len(files)} files, {total_size / (1024**2):.2f} MB")
            else:
                print(f"  • {name}: Not downloaded")


# ===========================
# 2. DATA PREPROCESSING
# ===========================

class UnsupervisedDataProcessor:
    """Implements unsupervised preprocessing from conference paper methodology"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load spaCy for NER and linguistic features
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except:
            print("⚠️  Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def load_race_dataset(self) -> List[Dict]:
        """Load RACE dataset (primary reading comprehension data)"""
        race_dir = self.data_dir / "race"
        passages = []
        
        print("\n📖 Loading RACE dataset...")
        print(f"   Looki ng in: {race_dir.absolute()}")
        
        # Check if directory exists
        if not race_dir.exists():
            print(f"⚠️  RACE directory not found. Skipping...")
            return passages
        
        # Try multiple possible structures
        # Structure 1: train/dev/test -> middle/high
        for split in ['train', 'dev', 'test']:
            split_dir = race_dir / split
            if split_dir.exists():
                for level in ['middle', 'high']:
                    level_dir = split_dir / level
                    if level_dir.exists():
                        for file_path in level_dir.glob("*.txt"):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    passages.append({
                                        'id': file_path.stem,
                                        'passage': data.get('article', ''),
                                        'questions': data.get('questions', []),
                                        'answers': data.get('answers', []),
                                        'options': data.get('options', []),
                                        'difficulty': level,
                                        'split': split
                                    })
                            except Exception as e:
                                print(f"⚠️  Error reading {file_path}: {e}")
                                continue
        
        # Structure 2: Direct .json files in race_dir
        for json_file in race_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both single object and array formats
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if 'article' in item or 'passage' in item or 'context' in item:
                            passages.append({
                                'id': json_file.stem,
                                'passage': item.get('article', item.get('passage', item.get('context', ''))),
                                'questions': item.get('questions', []),
                                'answers': item.get('answers', []),
                                'options': item.get('options', []),
                                'difficulty': item.get('difficulty', 'intermediate'),
                                'split': 'train'
                            })
            except Exception as e:
                continue
        
        print(f"✅ Loaded {len(passages)} passages from RACE")
        return passages
    
    def load_squad_dataset(self) -> List[Dict]:
        """Load SQuAD dataset"""
        squad_dir = self.data_dir / "squad"
        passages = []
        
        print("\n📖 Loading SQuAD dataset...")
        print(f"   Looking in: {squad_dir.absolute()}")
        
        if not squad_dir.exists():
            print(f"⚠️  SQuAD directory not found. Skipping...")
            return passages
        
        # Try all JSON files
        for json_file in squad_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # SQuAD structure: data -> paragraphs -> context + qas
                    if 'data' in data:
                        for article in data.get('data', []):
                            for paragraph in article.get('paragraphs', []):
                                context = paragraph.get('context', '')
                                qas = paragraph.get('qas', [])
                                
                                if context and qas:
                                    passages.append({
                                        'id': f"squad_{len(passages)}",
                                        'passage': context,
                                        'questions': [qa['question'] for qa in qas],
                                        'answers': [qa['answers'][0]['text'] if qa.get('answers') else '' for qa in qas],
                                        'difficulty': 'intermediate',
                                        'split': 'train'
                                    })
                    # Alternative: direct list of contexts
                    elif isinstance(data, list):
                        for item in data:
                            if 'context' in item or 'passage' in item:
                                passages.append({
                                    'id': f"squad_{len(passages)}",
                                    'passage': item.get('context', item.get('passage', '')),
                                    'questions': item.get('questions', []),
                                    'answers': item.get('answers', []),
                                    'difficulty': 'intermediate',
                                    'split': 'train'
                                })
            except Exception as e:
                print(f"⚠️  Error reading {json_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(passages)} passages from SQuAD")
        return passages
    
    def load_cefr_texts(self) -> List[Dict]:
        """Load CEFR levelled texts"""
        cefr_dir = self.data_dir / "cefr_texts"
        passages = []
        
        print("\n📖 Loading CEFR levelled texts...")
        print(f"   Looking in: {cefr_dir.absolute()}")
        
        if not cefr_dir.exists():
            print(f"⚠️  CEFR directory not found. Skipping...")
            return passages
        
        # Try CSV files
        for csv_file in cefr_dir.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    text = row.get('text', row.get('passage', row.get('content', '')))
                    if text and len(str(text).strip()) > 50:
                        passages.append({
                            'id': f"cefr_{len(passages)}",
                            'passage': str(text),
                            'questions': [],
                            'answers': [],
                            'difficulty': row.get('level', row.get('difficulty', 'B1')),
                            'split': 'train'
                        })
            except Exception as e:
                print(f"⚠️  Error reading {csv_file}: {e}")
                continue
        
        # Try JSON files
        for json_file in cefr_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        text = item.get('text', item.get('passage', item.get('content', '')))
                        if text and len(str(text).strip()) > 50:
                            passages.append({
                                'id': f"cefr_{len(passages)}",
                                'passage': str(text),
                                'questions': [],
                                'answers': [],
                                'difficulty': item.get('level', item.get('difficulty', 'B1')),
                                'split': 'train'
                            })
            except Exception as e:
                continue
        
        print(f"✅ Loaded {len(passages)} passages from CEFR")
        return passages
    
    def aggregate_text_corpus(self) -> List[Dict]:
        """Aggregate all datasets - implements paper's 'Aggregated Text Corpus' step"""
        print("\n🔄 Aggregating text corpus from multiple sources...")
        print("=" * 60)
        
        all_passages = []
        
        # Load each dataset
        all_passages.extend(self.load_race_dataset())
        all_passages.extend(self.load_squad_dataset())
        all_passages.extend(self.load_cefr_texts())
        
        # If no data loaded, create synthetic samples for testing
        if len(all_passages) == 0:
            print("\n⚠️  No datasets loaded. Generating synthetic data for testing...")
            all_passages = self._generate_synthetic_data(100)
        
        print(f"\n✅ Total aggregated passages: {len(all_passages)}")
        
        # Save aggregated corpus
        corpus_file = self.processed_dir / "aggregated_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(all_passages, f, indent=2)
        print(f"💾 Saved to: {corpus_file}")
        
        return all_passages
    
    def _generate_synthetic_data(self, n_samples: int = 100) -> List[Dict]:
        """Generate synthetic reading passages for testing when real data unavailable"""
        synthetic_passages = []
        
        templates = [
            "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas.",
            "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
            "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation. The cycling of water in and out of the atmosphere is a significant aspect of the weather patterns on Earth.",
            "Shakespeare wrote numerous plays during the Elizabethan era. His works include tragedies such as Hamlet and Romeo and Juliet, comedies like A Midsummer Night's Dream, and historical plays such as Henry V. His influence on English literature and language remains profound to this day.",
            "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches."
        ]
        
        difficulties = ['easy', 'intermediate', 'high']
        
        for i in range(n_samples):
            passage = templates[i % len(templates)]
            difficulty = difficulties[i % len(difficulties)]
            
            synthetic_passages.append({
                'id': f'synthetic_{i}',
                'passage': passage + f" Additional context for sample {i} to add variety.",
                'questions': [
                    f"What is the main topic discussed in this passage?",
                    f"According to the text, what is a key characteristic?",
                    f"How does this concept relate to real-world applications?"
                ],
                'answers': ['General summary', 'Key feature', 'Practical use'],
                'difficulty': difficulty,
                'split': 'train'
            })
        
        print(f"✅ Generated {len(synthetic_passages)} synthetic passages")
        return synthetic_passages
    
    def noun_entity_aware_masking(self, text: str, mask_ratio: float = 0.15) -> Tuple[str, List[str]]:
        """
        Implements Noun-Entity-Aware Masking from paper (Few-Shot MRC)
        Selectively masks nouns and named entities for unsupervised post-training
        """
        doc = self.nlp(text)
        
        # Identify noun and entity tokens
        maskable_tokens = []
        for token in doc:
            # Mask nouns (NOUN, PROPN) and named entities
            if token.pos_ in ['NOUN', 'PROPN'] or token.ent_type_:
                maskable_tokens.append(token)
        
        # Randomly select tokens to mask
        n_mask = max(1, int(len(maskable_tokens) * mask_ratio))
        if len(maskable_tokens) == 0:
            return text, []
        
        masked_indices = np.random.choice(len(maskable_tokens), n_mask, replace=False)
        tokens_to_mask = [maskable_tokens[i] for i in masked_indices]
        
        # Create masked text
        masked_text = text
        masked_words = []
        for token in sorted(tokens_to_mask, key=lambda t: t.idx, reverse=True):
            original_word = token.text
            masked_words.append(original_word)
            # Replace with [MASK] token
            masked_text = masked_text[:token.idx] + "[MASK]" + masked_text[token.idx + len(original_word):]
        
        return masked_text, masked_words
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features for SVM classification (from paper)"""
        doc = self.nlp(text)
        
        # Calculate readability metrics
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct]
        
        # Flesch-Kincaid Grade Level approximation
        syllables = sum(len(re.findall(r'[aeiouy]+', token.text.lower())) for token in words)
        avg_syllables_per_word = syllables / len(words) if words else 0
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        flesch_kincaid = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        
        features = {
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_unique_words': len(set(token.text.lower() for token in words)),
            'avg_word_length': np.mean([len(token.text) for token in words]) if words else 0,
            'avg_sentence_length': avg_words_per_sentence,
            'flesch_kincaid_grade': max(0, flesch_kincaid),
            'num_nouns': sum(1 for token in doc if token.pos_ == 'NOUN'),
            'num_verbs': sum(1 for token in doc if token.pos_ == 'VERB'),
            'num_adjectives': sum(1 for token in doc if token.pos_ == 'ADJ'),
            'num_entities': len(doc.ents),
            'lexical_diversity': len(set(token.text.lower() for token in words)) / len(words) if words else 0
        }
        
        return features
    
    def create_training_splits(self, passages: List[Dict], 
                              train_ratio: float = 0.70,
                              val_ratio: float = 0.15) -> Dict:
        """Create train/val/test splits (70/15/15 as per paper)"""
        np.random.shuffle(passages)
        
        n = len(passages)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': passages[:train_end],
            'validation': passages[train_end:val_end],
            'test': passages[val_end:]
        }
        
        print(f"\n📊 Data splits created:")
        print(f"  • Training: {len(splits['train'])} passages ({train_ratio*100}%)")
        print(f"  • Validation: {len(splits['validation'])} passages ({val_ratio*100}%)")
        print(f"  • Test: {len(splits['test'])} passages ({100-train_ratio*100-val_ratio*100}%)")
        
        return splits
    
    def process_for_unsupervised_training(self, passages: List[Dict]) -> List[Dict]:
        """
        Process passages for unsupervised post-training
        Implements: Text Processing -> Unsupervised Post Training (from diagram)
        """
        print("\n🔬 Processing data for unsupervised post-training...")
        print("=" * 60)
        
        processed_data = []
        
        for i, passage in enumerate(passages):
            if i % 100 == 0:
                print(f"Processing passage {i}/{len(passages)}...")
            
            text = passage['passage']
            if not text or len(text.strip()) < 50:
                continue
            
            # Apply noun-entity-aware masking
            masked_text, masked_words = self.noun_entity_aware_masking(text)
            
            # Extract linguistic features for SVM
            features = self.extract_linguistic_features(text)
            
            processed_data.append({
                'id': passage['id'],
                'original_text': text,
                'masked_text': masked_text,
                'masked_words': masked_words,
                'questions': passage.get('questions', []),
                'answers': passage.get('answers', []),
                'difficulty': passage.get('difficulty', 'intermediate'),
                'linguistic_features': features
            })
        
        print(f"\n✅ Processed {len(processed_data)} passages with noun-entity masking")
        
        # Save processed data
        output_file = self.processed_dir / "unsupervised_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        print(f"💾 Saved to: {output_file}")
        
        return processed_data


# ===========================
# 3. FEATURE EXTRACTION FOR SVM
# ===========================

class SVMFeatureExtractor:
    """Extracts features for SVM difficulty classification (99% accuracy target)"""
    
    def __init__(self):
        self.difficulty_mapping = {
            'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6,
            'easy': 2, 'middle': 3, 'intermediate': 4, 'high': 5, 'advanced': 6
        }
    
    def prepare_svm_dataset(self, processed_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels for SVM training"""
        features_list = []
        labels = []
        
        print(f"Processing {len(processed_data)} items for SVM...")
        
        for item in processed_data:
            features = item['linguistic_features']
            feature_vector = [
                features['num_sentences'],
                features['num_words'],
                features['num_unique_words'],
                features['avg_word_length'],
                features['avg_sentence_length'],
                features['flesch_kincaid_grade'],
                features['num_nouns'],
                features['num_verbs'],
                features['num_adjectives'],
                features['num_entities'],
                features['lexical_diversity']
            ]
            features_list.append(feature_vector)
            
            # Map difficulty to numeric label
            difficulty = str(item['difficulty']).lower()
            label = self.difficulty_mapping.get(difficulty, 3)
            labels.append(label)
        
        X = np.array(features_list, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)  # Force integer type
        
        print(f"\n📊 SVM Feature Matrix: {X.shape}")
        print(f"📊 Labels distribution: {np.bincount(y)}")
        print(f"📊 Difficulty levels found: {np.unique(y)}")
        
        # Save for model training
        np.save('processed_data/svm_features.npy', X)
        np.save('processed_data/svm_labels.npy', y)
        
        return X, y


# ===========================
# 4. MAIN EXECUTION
# ===========================

def main():
    print("=" * 60)
    print("READAPT ADAPTIVE LEARNING SYSTEM")
    print("Data Collection & Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Download datasets from Kaggle
    print("\n[STEP 1] Data Acquisition from Kaggle API")
    print("Note: This step requires Kaggle API setup. Skipping if not configured...")
    try:
        collector = KaggleDataCollector()
        collector.download_datasets()
    except Exception as e:
        print(f"⚠️  Kaggle download skipped: {e}")
        print("Will proceed with any existing datasets or generate synthetic data")
    
    # Step 2: Process and aggregate data
    print("\n[STEP 2] Data Preprocessing & Aggregation")
    processor = UnsupervisedDataProcessor()
    
    # Aggregate corpus
    all_passages = processor.aggregate_text_corpus()
    
    if len(all_passages) == 0:
        print("\n❌ No data available. Please check dataset downloads.")
        return
    
    # Create splits
    splits = processor.create_training_splits(all_passages)
    
    # Process for unsupervised training
    print("\n🔄 Processing training data...")
    processed_train = processor.process_for_unsupervised_training(splits['train'])
    
    print("\n🔄 Processing validation data...")
    processed_val = processor.process_for_unsupervised_training(splits['validation'])
    
    print("\n🔄 Processing test data...")
    processed_test = processor.process_for_unsupervised_training(splits['test'])
    
    # Save splits separately
    for split_name, split_data in [('train', processed_train), ('val', processed_val), ('test', processed_test)]:
        split_file = processor.processed_dir / f"unsupervised_{split_name}_data.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
        print(f"💾 Saved {split_name} split: {len(split_data)} samples")
    
    # Step 3: Extract features for SVM
    print("\n[STEP 3] Feature Extraction for SVM Classification")
    svm_extractor = SVMFeatureExtractor()
    
    X_train, y_train = svm_extractor.prepare_svm_dataset(processed_train)
    X_val, y_val = svm_extractor.prepare_svm_dataset(processed_val)
    X_test, y_test = svm_extractor.prepare_svm_dataset(processed_test)
    
    # Save splits separately
    np.save('processed_data/svm_features_train.npy', X_train)
    np.save('processed_data/svm_labels_train.npy', y_train)
    np.save('processed_data/svm_features_val.npy', X_val)
    np.save('processed_data/svm_labels_val.npy', y_val)
    np.save('processed_data/svm_features_test.npy', X_test)
    np.save('processed_data/svm_labels_test.npy', y_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📁 All data saved to: processed_data/")
    print(f"📊 Training samples: {len(processed_train)}")
    print(f"📊 Validation samples: {len(processed_val)}")
    print(f"📊 Test samples: {len(processed_test)}")
    print("\n📈 Feature Matrix Shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    print("\nNext steps:")
    print("  1. Run model_training.py to train GPT-2 and SVM")
    print("  2. Evaluate on test set")
    print("  3. Integrate with Flask app")


if __name__ == "__main__":
    main()