import os
import json
import random
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any


class ShareGPTDataset:
    """
    Dataset loader for ShareGPT-style conversation data.
    Supports JSONL format with conversation history.
    """
    
    def __init__(self, data_path=None, train_size=2000, valid_size=200, test_size=200, 
                 format_type="chatml", split_ratio=0.8):
        self.data_path = data_path
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.format_type = format_type  # "chatml" or "instruction"
        self.split_ratio = split_ratio
        self.datasets = {'train': {}, 'valid': {}, 'test': {}}
    
    def load_sharegpt_data(self):
        """Load ShareGPT-style data from JSONL file"""
        raw_data = []
        
        # Handle both single file and directory
        if os.path.isfile(self.data_path):
            files = [self.data_path]
        else:
            files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) 
                    if f.endswith('.jsonl') or f.endswith('.json')]
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            raw_data.append(data)
                        except json.JSONDecodeError:
                            continue
        
        print(f"Loaded {len(raw_data)} conversations from {len(files)} file(s)")
        return raw_data
    
    def process_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process a single conversation into training examples"""
        processed_examples = []
        
        # Handle different ShareGPT formats
        if "conversations" in conversation:
            # Standard ShareGPT format
            convs = conversation["conversations"]
        elif "messages" in conversation:
            # Alternative format
            convs = conversation["messages"]
        else:
            # Assume the conversation itself is the messages
            convs = conversation if isinstance(conversation, list) else [conversation]
        
        # Process conversation pairs
        for i in range(0, len(convs) - 1, 2):
            if i + 1 < len(convs):
                user_msg = convs[i]
                assistant_msg = convs[i + 1]
                
                # Extract content based on format
                if isinstance(user_msg, dict):
                    user_content = user_msg.get("content", user_msg.get("value", ""))
                    assistant_content = assistant_msg.get("content", assistant_msg.get("value", ""))
                else:
                    user_content = str(user_msg)
                    assistant_content = str(assistant_msg)
                
                if user_content and assistant_content:
                    if self.format_type == "chatml":
                        text = self.format_chatml(user_content, assistant_content)
                    else:
                        text = self.format_instruction(user_content, assistant_content)
                    
                    processed_examples.append({"text": text})
        
        return processed_examples
    
    def format_chatml(self, user_content: str, assistant_content: str) -> str:
        """Format as ChatML conversation"""
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
    
    def format_instruction(self, user_content: str, assistant_content: str) -> str:
        """Format as instruction-response pair"""
        return f"### Instruction:\n{user_content}\n\n### Response:\n{assistant_content}"
    
    def create_splits(self, all_examples: List[Dict[str, str]]):
        """Create train/valid/test splits"""
        random.shuffle(all_examples)
        
        total = len(all_examples)
        train_end = int(total * self.split_ratio)
        valid_end = train_end + int(total * 0.1)
        
        train_data = all_examples[:train_end]
        valid_data = all_examples[train_end:valid_end]
        test_data = all_examples[valid_end:]
        
        # Apply size limits
        if self.train_size > 0 and len(train_data) > self.train_size:
            train_data = random.sample(train_data, self.train_size)
        
        if self.valid_size > 0 and len(valid_data) > self.valid_size:
            valid_data = random.sample(valid_data, self.valid_size)
        
        if self.test_size > 0 and len(test_data) > self.test_size:
            test_data = random.sample(test_data, self.test_size)
        
        return train_data, valid_data, test_data
    
    def load_data(self):
        """Main method to load and process ShareGPT data"""
        print("Loading ShareGPT data...")
        
        # Load raw conversations
        raw_conversations = self.load_sharegpt_data()
        
        # Process all conversations
        all_examples = []
        for conv in raw_conversations:
            examples = self.process_conversation(conv)
            all_examples.extend(examples)
        
        print(f"Processed {len(all_examples)} training examples")
        
        # Create splits
        train_data, valid_data, test_data = self.create_splits(all_examples)
        
        # Convert to datasets
        self.datasets['train'] = Dataset.from_pandas(pd.DataFrame(train_data))
        self.datasets['valid'] = Dataset.from_pandas(pd.DataFrame(valid_data))
        self.datasets['test'] = Dataset.from_pandas(pd.DataFrame(test_data))
        
        print(f"Created splits - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
        
        return self.datasets


class ShareGPTDatasetFromFile(ShareGPTDataset):
    """
    Alternative dataset loader that works with the existing JoLADataset interface
    """
    
    def __init__(self, data_path=None, train_size=2000, format_type="chatml"):
        super().__init__(data_path=data_path, train_size=train_size, format_type=format_type)
        self.jola_datasets = {'train': {}, 'valid': {}, 'test': {}}
    
    def data_from_file(self):
        """Compatible with existing JoLADataset interface"""
        return self.load_data()


def create_sharegpt_dataset(data_path, train_size=2000, format_type="chatml"):
    """
    Convenience function to create ShareGPT dataset
    """
    dataset = ShareGPTDataset(
        data_path=data_path,
        train_size=train_size,
        format_type=format_type
    )
    return dataset.load_data()
