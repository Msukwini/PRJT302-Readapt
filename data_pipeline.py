"""
Readapt Adaptive Learning System - Data Collection & Processing Pipeline
FIXED: Properly loads REAL Kaggle datasets with UNIFIED difficulty levels
"""

import os
import json
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
        if not folder_path.is_dir():
            print(f"Error: Path is not an existing directory at '{folder_path_str}'")
            return False
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
        """
            'mmlu': 'peiyuanliu2001/mmlu-dataset',
            'en_cn_learning': 'datasetengineer/english-chinese-learning-dataset',
            'fasttext_news': 'facebook/fasttext-wikinews'
        }
        """
        # Dataset configurations
        self.datasets = {
            'race': 'ankitdhiman7/race-dataset',
            'cefr_texts': 'amontgomerie/cefr-levelled-english-texts',
            'iot_learning': 'ziya07/language-learning-analysis-with-iot',
            'squad': 'stanfordu/stanford-question-answering-dataset',
        }
    
    def setup_kaggle_api(self):
        """Initialize Kaggle API - requires kaggle.json in ~/.kaggle/"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("âœ… Kaggle API authenticated successfully")
            return api
        except Exception as e:
            print(f"âŒ Kaggle API setup failed: {e}")
            print("Please ensure kaggle.json is in ~/.kaggle/ directory")
            return None
    
    def download_datasets(self):
        """Download all configured datasets"""
        api = self.setup_kaggle_api()
        if not api:
            return False
        
        print(f"\nðŸ“¥ Starting dataset downloads to: {self.base_dir.absolute()}")
        print("=" * 60)
        
        for name, dataset_path in self.datasets.items():
            download_dir = self.base_dir / name
            download_dir.mkdir(exist_ok=True)
            
            try:
                print(f"\nâ³ Downloading {name}: {dataset_path}")
                api.dataset_download_files(
                    dataset_path,
                    path=download_dir,
                    unzip=True,
                    quiet=False
                )
                print(f"âœ… {name} downloaded successfully")
                
            except Exception as e:
                print(f"âš ï¸  Failed to download {name}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("ðŸ“Š Download Summary:")
        self._print_dataset_summary()
        return True
    
    def _print_dataset_summary(self):
        """Print summary of downloaded datasets"""
        for name in self.datasets.keys():
            dataset_dir = self.base_dir / name
            if dataset_dir.exists():
                files = list(dataset_dir.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  â€¢ {name}: {len(files)} files, {total_size / (1024**2):.2f} MB")
            else:
                print(f"  â€¢ {name}: Not downloaded")


class UnsupervisedDataProcessor:
    """Implements unsupervised preprocessing with UNIFIED difficulty mapping"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
        # UNIFIED DIFFICULTY MAPPING (1-6 scale)
        # Maps all dataset-specific difficulties to standardized 1-6 scale
        self.difficulty_unified_map = {
            # RACE dataset: middle/high
            'middle': 3,      # Intermediate
            'high': 5,        # Advanced
            
            # CEFR dataset: A1, A2, B1, B2, C1, C2
            'a1': 1,          # Beginner
            'a2': 2,          # Elementary
            'b1': 3,          # Intermediate
            'b2': 4,          # Upper Intermediate
            'c1': 5,          # Advanced
            'c2': 6,          # Expert
            
            # Generic labels (fallback)
            'easy': 2,
            'beginner': 1,
            'elementary': 2,
            'intermediate': 3,
            'upper intermediate': 4,
            'advanced': 5,
            'expert': 6,
            'hard': 5,
            'difficult': 5,
            
            # Default for unlabeled
            'unknown': 3,
            'general': 3
        }
        
        # Load spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except:
            print("⚠️  Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def normalize_difficulty(self, raw_difficulty: str) -> int:
        """
        Normalize any difficulty label to unified 1-6 scale
        """
        if isinstance(raw_difficulty, (int, float)):
            # Already numeric, clamp to 1-6
            return max(1, min(6, int(raw_difficulty)))
        
        # Convert to lowercase and strip
        diff_str = str(raw_difficulty).lower().strip()
        
        # Look up in unified map
        unified = self.difficulty_unified_map.get(diff_str, 3)  # Default to 3 (intermediate)
        
        return unified
    
    def load_race_dataset(self) -> List[Dict]:
        """
        Load RACE dataset - FIXED VERSION
        Properly handles CSV format with article, question, options, answer columns
        """
        race_dir = self.data_dir / "race"
        passages = []
        
        print("\n📖 Loading RACE dataset...")
        print(f"   Looking in: {race_dir.absolute()}")
        
        if not race_dir.exists():
            print(f"⚠️  RACE directory not found. Skipping...")
            return passages
        
        # RACE dataset comes as CSV files: train.csv, dev.csv, test.csv
        csv_files = list(race_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"⚠️  No CSV files found in RACE directory")
            return passages
        
        print(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                print(f"   Processing: {csv_file.name}")
                df = pd.read_csv(csv_file)
                
                print(f"      Columns: {list(df.columns)}")
                print(f"      Rows: {len(df)}")
                
                # Expected columns: id, article, question, A, B, C, D, answer
                required_cols = ['article']
                if not all(col in df.columns for col in required_cols):
                    print(f"      ⚠️  Missing required columns, skipping")
                    continue
                
                # Group by article (each article may have multiple questions)
                articles_seen = set()
                
                for idx, row in df.iterrows():
                    article = str(row.get('article', '')).strip()
                    
                    if not article or len(article) < 50:
                        continue
                    
                    # Avoid duplicate articles
                    if article in articles_seen:
                        continue
                    articles_seen.add(article)
                    
                    # Extract questions for this article (if available)
                    questions = []
                    answers = []
                    options = []
                    
                    if 'question' in df.columns:
                        # Get all rows with this article
                        article_rows = df[df['article'] == article]
                        for _, q_row in article_rows.iterrows():
                            q = str(q_row.get('question', '')).strip()
                            if q:
                                questions.append(q)
                                
                                # Get answer
                                ans = str(q_row.get('answer', '')).strip()
                                answers.append(ans)
                                
                                # Get options (A, B, C, D)
                                opts = []
                                for opt_col in ['A', 'B', 'C', 'D']:
                                    if opt_col in q_row:
                                        opts.append(str(q_row[opt_col]).strip())
                                options.append(opts)
                    
                    # Determine difficulty from filename or default to 'middle'
                    # RACE typically has 'middle' and 'high' school levels
                    if 'high' in csv_file.stem.lower():
                        difficulty = 'high'
                    elif 'middle' in csv_file.stem.lower():
                        difficulty = 'middle'
                    else:
                        # Use article length as heuristic
                        difficulty = 'high' if len(article) > 500 else 'middle'
                    
                    passages.append({
                        'id': f'race_{len(passages)}',
                        'passage': article,
                        'questions': questions,
                        'answers': answers,
                        'options': options,
                        'difficulty': difficulty,  # Will be normalized later
                        'split': csv_file.stem,  # train/dev/test
                        'source': 'RACE'
                    })
                
                print(f"      ✅ Extracted {len([p for p in passages if p['source'] == 'RACE' and p['split'] == csv_file.stem])} unique articles")
                
            except Exception as e:
                print(f"      ❌ Error reading {csv_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(passages)} passages from RACE")
        return passages
    
    def load_cefr_texts(self) -> List[Dict]:
        """
        Load CEFR levelled texts - FIXED VERSION
        CSV with columns: text, label (A1, A2, B1, B2, C1, C2)
        """
        cefr_dir = self.data_dir / "cefr_texts"
        passages = []
        
        print("\n📖 Loading CEFR levelled texts...")
        print(f"   Looking in: {cefr_dir.absolute()}")
        
        if not cefr_dir.exists():
            print(f"⚠️  CEFR directory not found. Skipping...")
            return passages
        
        # Look for CSV file
        csv_files = list(cefr_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"⚠️  No CSV files found in CEFR directory")
            return passages
        
        for csv_file in csv_files:
            try:
                print(f"   Processing: {csv_file.name}")
                df = pd.read_csv(csv_file)
                
                print(f"      Columns: {list(df.columns)}")
                print(f"      Rows: {len(df)}")
                
                # Expected columns: text, label (or level)
                text_col = 'text' if 'text' in df.columns else 'passage'
                level_col = 'label' if 'label' in df.columns else 'level'
                
                if text_col not in df.columns:
                    print(f"      ⚠️  No text column found, skipping")
                    continue
                
                for idx, row in df.iterrows():
                    text = str(row.get(text_col, '')).strip()
                    
                    if not text or len(text) < 50:
                        continue
                    
                    # Get CEFR level (A1, A2, B1, B2, C1, C2)
                    level = str(row.get(level_col, 'B1')).strip()
                    
                    passages.append({
                        'id': f'cefr_{len(passages)}',
                        'passage': text,
                        'questions': [],
                        'answers': [],
                        'options': [],
                        'difficulty': level,  # Will be normalized (A1=1, B1=3, C2=6)
                        'split': 'train',
                        'source': 'CEFR'
                    })
                
                print(f"      ✅ Extracted {len([p for p in passages if p['source'] == 'CEFR'])} texts")
                
            except Exception as e:
                print(f"      ❌ Error reading {csv_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(passages)} passages from CEFR")
        return passages
    
    def load_iot_learning_dataset(self) -> List[Dict]:
        """
        Load IoT Learning Analysis dataset
        Contains student performance data with reading activities
        """
        iot_dir = self.data_dir / "iot_learning"
        passages = []
        
        print("\n📖 Loading IoT Learning dataset...")
        print(f"   Looking in: {iot_dir.absolute()}")
        
        if not iot_dir.exists():
            print(f"⚠️  IoT Learning directory not found. Skipping...")
            return passages
        
        csv_files = list(iot_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"⚠️  No CSV files found in IoT Learning directory")
            return passages
        
        for csv_file in csv_files:
            try:
                print(f"   Processing: {csv_file.name}")
                df = pd.read_csv(csv_file)
                
                print(f"      Columns: {list(df.columns)}")
                
                # This dataset has activity_score, so we can infer difficulty
                # High scores → text was easier; Low scores → text was harder
                
                # Look for text-related columns
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'passage' in col.lower() or 'content' in col.lower()]
                
                if not text_cols:
                    print(f"      ⚠️  No text columns found, skipping")
                    continue
                
                text_col = text_cols[0]
                
                for idx, row in df.iterrows():
                    text = str(row.get(text_col, '')).strip()
                    
                    if not text or len(text) < 50:
                        continue
                    
                    # Infer difficulty from activity_score if available
                    if 'activity_score' in df.columns:
                        score = row.get('activity_score', 75)
                        # Lower scores suggest harder text
                        if score >= 85:
                            difficulty = 'easy'
                        elif score >= 70:
                            difficulty = 'intermediate'
                        else:
                            difficulty = 'advanced'
                    else:
                        difficulty = 'intermediate'
                    
                    passages.append({
                        'id': f'iot_{len(passages)}',
                        'passage': text,
                        'questions': [],
                        'answers': [],
                        'options': [],
                        'difficulty': difficulty,
                        'split': 'train',
                        'source': 'IoT_Learning'
                    })
                
                print(f"      ✅ Extracted {len([p for p in passages if p['source'] == 'IoT_Learning'])} texts")
                
            except Exception as e:
                print(f"      ❌ Error reading {csv_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(passages)} passages from IoT Learning")
        return passages
    
    def load_squad_dataset(self) -> List[Dict]:
        """
        Load SQuAD dataset
        JSON format with context + questions
        """
        squad_dir = self.data_dir / "squad"
        passages = []
        
        print("\n📖 Loading SQuAD dataset...")
        print(f"   Looking in: {squad_dir.absolute()}")
        
        if not squad_dir.exists():
            print(f"⚠️  SQuAD directory not found. Skipping...")
            return passages
        
        json_files = list(squad_dir.glob("**/*.json"))
        
        if not json_files:
            print(f"⚠️  No JSON files found in SQuAD directory")
            return passages
        
        for json_file in json_files:
            try:
                print(f"   Processing: {json_file.name}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # SQuAD structure: data -> paragraphs -> context + qas
                if 'data' in data:
                    for article in data.get('data', []):
                        for paragraph in article.get('paragraphs', []):
                            context = paragraph.get('context', '').strip()
                            qas = paragraph.get('qas', [])
                            
                            if not context or len(context) < 50:
                                continue
                            
                            questions = []
                            answers = []
                            
                            for qa in qas:
                                q = qa.get('question', '').strip()
                                if q:
                                    questions.append(q)
                                    
                                    # Get answer
                                    ans_list = qa.get('answers', [])
                                    if ans_list:
                                        answers.append(ans_list[0].get('text', ''))
                                    else:
                                        answers.append('')
                            
                            # SQuAD is intermediate difficulty
                            passages.append({
                                'id': f'squad_{len(passages)}',
                                'passage': context,
                                'questions': questions,
                                'answers': answers,
                                'options': [],
                                'difficulty': 'intermediate',  # SQuAD is generally intermediate
                                'split': 'train',
                                'source': 'SQuAD'
                            })
                
                print(f"      ✅ Extracted {len([p for p in passages if p['source'] == 'SQuAD'])} contexts")
                
            except Exception as e:
                print(f"      ❌ Error reading {json_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(passages)} passages from SQuAD")
        return passages
    
    def aggregate_text_corpus(self) -> List[Dict]:
        """
        Aggregate all datasets with UNIFIED difficulty levels
        """
        print("\n🔄 Aggregating text corpus from REAL datasets...")
        print("=" * 60)
        
        all_passages = []
        
        # Load each dataset
        race_data = self.load_race_dataset()
        cefr_data = self.load_cefr_texts()
        iot_data = self.load_iot_learning_dataset()
        squad_data = self.load_squad_dataset()
        
        all_passages.extend(race_data)
        all_passages.extend(cefr_data)
        all_passages.extend(iot_data)
        all_passages.extend(squad_data)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   RACE: {len(race_data)} passages")
        print(f"   CEFR: {len(cefr_data)} passages")
        print(f"   IoT Learning: {len(iot_data)} passages")
        print(f"   SQuAD: {len(squad_data)} passages")
        print(f"   Total: {len(all_passages)} passages")
        
        if len(all_passages) == 0:
            print("\n❌ No data loaded from any dataset!")
            print("   Please check that datasets are downloaded to the 'datasets/' folder")
            return []
        
        # NORMALIZE all difficulties to 1-6 scale
        print(f"\n🔄 Normalizing difficulty levels to unified 1-6 scale...")
        
        difficulty_distribution = defaultdict(int)
        
        for passage in all_passages:
            raw_difficulty = passage['difficulty']
            unified_difficulty = self.normalize_difficulty(raw_difficulty)
            passage['difficulty_original'] = raw_difficulty  # Keep original for reference
            passage['difficulty'] = unified_difficulty  # Overwrite with unified (1-6)
            difficulty_distribution[unified_difficulty] += 1
        
        print(f"\n✅ Unified difficulty distribution:")
        for level in sorted(difficulty_distribution.keys()):
            count = difficulty_distribution[level]
            pct = count / len(all_passages) * 100
            print(f"   Level {level}: {count} passages ({pct:.1f}%)")
        
        # Verify we have multiple classes
        unique_difficulties = set(p['difficulty'] for p in all_passages)
        print(f"\n✅ Unique difficulty levels: {sorted(unique_difficulties)}")
        
        if len(unique_difficulties) < 2:
            print(f"\n⚠️  WARNING: Only {len(unique_difficulties)} difficulty level(s) found!")
            print("   SVM requires at least 2 classes to train.")
        
        # Save aggregated corpus
        corpus_file = self.processed_dir / "aggregated_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(all_passages, f, indent=2)
        print(f"\n💾 Saved aggregated corpus to: {corpus_file}")
        
        return all_passages
    
    def noun_entity_aware_masking(self, text: str, mask_ratio: float = 0.15) -> Tuple[str, List[str]]:
        """Implements Noun-Entity-Aware Masking from paper"""
        doc = self.nlp(text)
        
        maskable_tokens = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] or token.ent_type_:
                maskable_tokens.append(token)
        
        if len(maskable_tokens) == 0:
            return text, []
        
        n_mask = max(1, int(len(maskable_tokens) * mask_ratio))
        masked_indices = np.random.choice(len(maskable_tokens), min(n_mask, len(maskable_tokens)), replace=False)
        tokens_to_mask = [maskable_tokens[i] for i in masked_indices]
        
        masked_text = text
        masked_words = []
        for token in sorted(tokens_to_mask, key=lambda t: t.idx, reverse=True):
            original_word = token.text
            masked_words.append(original_word)
            masked_text = masked_text[:token.idx] + "[MASK]" + masked_text[token.idx + len(original_word):]
        
        return masked_text, masked_words
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features for SVM classification"""
        doc = self.nlp(text)
        
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct]
        
        if len(words) == 0:
            return {
                'num_sentences': 0, 'num_words': 0, 'num_unique_words': 0,
                'avg_word_length': 0, 'avg_sentence_length': 0, 'flesch_kincaid_grade': 0,
                'num_nouns': 0, 'num_verbs': 0, 'num_adjectives': 0,
                'num_entities': 0, 'lexical_diversity': 0
            }
        
        syllables = sum(len(re.findall(r'[aeiouy]+', token.text.lower())) for token in words)
        avg_syllables_per_word = syllables / len(words)
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        flesch_kincaid = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        
        features = {
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_unique_words': len(set(token.text.lower() for token in words)),
            'avg_word_length': np.mean([len(token.text) for token in words]),
            'avg_sentence_length': avg_words_per_sentence,
            'flesch_kincaid_grade': max(0, flesch_kincaid),
            'num_nouns': sum(1 for token in doc if token.pos_ == 'NOUN'),
            'num_verbs': sum(1 for token in doc if token.pos_ == 'VERB'),
            'num_adjectives': sum(1 for token in doc if token.pos_ == 'ADJ'),
            'num_entities': len(doc.ents),
            'lexical_diversity': len(set(token.text.lower() for token in words)) / len(words)
        }
        
        return features
    
    def create_training_splits(self, passages: List[Dict], 
                              train_ratio: float = 0.70,
                              val_ratio: float = 0.15) -> Dict:
        """Create STRATIFIED train/val/test splits to ensure class balance"""
        
        # Stratify by difficulty
        by_difficulty = defaultdict(list)
        for p in passages:
            by_difficulty[p['difficulty']].append(p)
        
        train_split = []
        val_split = []
        test_split = []
        
        print(f"\n📊 Creating stratified splits:")
        for difficulty, items in sorted(by_difficulty.items()):
            np.random.shuffle(items)
            n = len(items)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_split.extend(items[:train_end])
            val_split.extend(items[train_end:val_end])
            test_split.extend(items[val_end:])
            
            print(f"   Level {difficulty}: {train_end} train, {val_end-train_end} val, {n-val_end} test")
        
        # Shuffle each split
        np.random.shuffle(train_split)
        np.random.shuffle(val_split)
        np.random.shuffle(test_split)
        
        splits = {
            'train': train_split,
            'validation': val_split,
            'test': test_split
        }
        
        print(f"\n📊 Final split sizes:")
        print(f"  • Training: {len(train_split)} passages")
        print(f"  • Validation: {len(val_split)} passages")
        print(f"  • Test: {len(test_split)} passages")
        
        return splits
    
    def process_for_unsupervised_training(self, passages: List[Dict]) -> List[Dict]:
        """Process passages for unsupervised post-training"""
        print(f"\n🔬 Processing {len(passages)} passages...")
        
        processed_data = []
        
        for i, passage in enumerate(passages):
            if i % 100 == 0 and i > 0:
                print(f"   Processed {i}/{len(passages)}...")
            
            text = passage['passage']
            if not text or len(text.strip()) < 50:
                continue
            
            masked_text, masked_words = self.noun_entity_aware_masking(text)
            features = self.extract_linguistic_features(text)
            
            processed_data.append({
                'id': passage['id'],
                'original_text': text,
                'masked_text': masked_text,
                'masked_words': masked_words,
                'questions': passage.get('questions', []),
                'answers': passage.get('answers', []),
                'difficulty': passage['difficulty'],  # Unified 1-6
                'linguistic_features': features,
                'source': passage.get('source', 'unknown')
            })
        
        print(f"✅ Processed {len(processed_data)} passages")
        return processed_data


class SVMFeatureExtractor:
    """Extracts features for SVM difficulty classification"""
    
    def prepare_svm_dataset(self, processed_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels for SVM training"""
        features_list = []
        labels = []
        
        print(f"\n📊 Preparing SVM dataset from {len(processed_data)} items...")
        
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
            labels.append(item['difficulty'])  # Already unified 1-6
        
        X = np.array(features_list, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        
        print(f"\n📊 SVM Feature Matrix: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"📊 Difficulty distribution:")
        for level, count in zip(unique, counts):
            print(f"   Level {level}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y


def main():
    print("=" * 60)
    print("READAPT DATA PIPELINE - REAL KAGGLE DATASETS")
    print("=" * 60)

    # Step 1: Download datasets from Kaggle
    print("\n[STEP 1] Data Acquisition from Kaggle API")
    print("Note: This step requires Kaggle API setup. Skipping if not configured...")
    try:
        collector = KaggleDataCollector()
        collector.download_datasets()
    except Exception as e:
        print(f"âš ï¸  Kaggle download skipped: {e}")
        print("Will proceed with any existing datasets or generate synthetic data")
    
    # Step 2: Process and aggregate data
    print("\n[STEP 2] Data Preprocessing & Aggregation")
    processor = UnsupervisedDataProcessor()
    
    # Aggregate corpus from REAL datasets
    all_passages = processor.aggregate_text_corpus()
    
    if len(all_passages) == 0:
        print("\n❌ No data loaded. Please download datasets to 'datasets/' folder")
        return
    
    # Verify multiple classes
    unique_diffs = set(p['difficulty'] for p in all_passages)
    if len(unique_diffs) < 2:
        print(f"\n❌ ERROR: Only {len(unique_diffs)} difficulty level(s) found!")
        return
    
    # Create stratified splits
    splits = processor.create_training_splits(all_passages)
    
    # Process each split
    print("\n🔄 Processing training data...")
    processed_train = processor.process_for_unsupervised_training(splits['train'])
    
    print("\n🔄 Processing validation data...")
    processed_val = processor.process_for_unsupervised_training(splits['validation'])
    
    print("\n🔄 Processing test data...")
    processed_test = processor.process_for_unsupervised_training(splits['test'])
    
    # Save splits
    for split_name, split_data in [('train', processed_train), ('val', processed_val), ('test', processed_test)]:
        split_file = processor.processed_dir / f"unsupervised_{split_name}_data.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
        print(f"💾 Saved {split_name}: {len(split_data)} samples")
    
    # Extract features for SVM
    print("\n[STEP 3] Feature Extraction for SVM")
    svm_extractor = SVMFeatureExtractor()
    
    X_train, y_train = svm_extractor.prepare_svm_dataset(processed_train)
    X_val, y_val = svm_extractor.prepare_svm_dataset(processed_val)
    X_test, y_test = svm_extractor.prepare_svm_dataset(processed_test)
    
    # Final verification
    print(f"\n🔍 Final Verification:")
    print(f"   Train classes: {len(np.unique(y_train))} unique difficulties")
    print(f"   Val classes: {len(np.unique(y_val))} unique difficulties")
    print(f"   Test classes: {len(np.unique(y_test))} unique difficulties")
    
    if len(np.unique(y_train)) < 2:
        print(f"\n❌ CRITICAL: Training set only has {len(np.unique(y_train))} class(es)!")
        return
    
    # Save features
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
    print(f"\n📈 Feature Matrix Shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"\n✅ Data ready for model training!")
    print("\nNext step: Run 'python model_training.py'")


if __name__ == "__main__":
    main()