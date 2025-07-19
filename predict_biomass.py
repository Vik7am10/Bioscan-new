#!/usr/bin/env python3
"""
Biomass Prediction Script - Load trained model and predict biomass from specimen images
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models

class SimpleBiomassEstimator(nn.Module):
    """Simple CNN for biomass estimation."""
    def __init__(self):
        super().__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        return self.backbone(x)

class BiomassPredictor:
    """
    Biomass prediction class using trained ResNet model
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize the biomass predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        print(f"🔄 Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        
        # Define image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model"""
        # Create model architecture (same as training)
        model = SimpleBiomassEstimator()
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def predict_single_image(self, image_path: str) -> float:
        """
        Predict biomass for a single image
        
        Args:
            image_path: Path to the specimen image
            
        Returns:
            Predicted biomass weight in mg
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(image_tensor)
                biomass_mg = prediction.item()
                
            return biomass_mg
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict biomass for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"🔄 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(image_paths)} images...")
                
            prediction = self.predict_single_image(image_path)
            
            result = {
                'image_path': image_path,
                'specimen_id': os.path.basename(image_path).replace('.jpg', ''),
                'predicted_biomass_mg': prediction
            }
            results.append(result)
            
        return results
    
    def test_on_validation_set(self, bin_mapping_csv: str, image_index_json: str, num_samples: int = 50):
        """
        Test predictions on validation samples with known biomass values
        
        Args:
            bin_mapping_csv: Path to BIN mapping CSV
            image_index_json: Path to image index JSON
            num_samples: Number of samples to test
        """
        print(f"🔄 Testing on {num_samples} validation samples...")
        
        # Load data
        with open(image_index_json, 'r') as f:
            image_index = json.load(f)
        
        bin_df = pd.read_csv(bin_mapping_csv)
        
        # Get validation samples
        test_samples = []
        for _, row in bin_df.iterrows():
            try:
                if isinstance(row['bioscan_processids'], str):
                    bioscan_processids = eval(row['bioscan_processids'])
                else:
                    bioscan_processids = row['bioscan_processids']
                
                for processid in bioscan_processids:
                    if processid in image_index:
                        image_path = image_index[processid]
                        if os.path.exists(image_path):
                            test_samples.append({
                                'image_path': image_path,
                                'specimen_id': processid,
                                'true_biomass_mg': row['mean_weight'],
                                'bin_id': row['dna_bin']
                            })
                            
                            if len(test_samples) >= num_samples:
                                break
            except:
                continue
                
            if len(test_samples) >= num_samples:
                break
        
        # Make predictions
        print(f"✅ Testing on {len(test_samples)} samples...")
        
        results = []
        for sample in test_samples:
            prediction = self.predict_single_image(sample['image_path'])
            
            result = {
                'specimen_id': sample['specimen_id'],
                'bin_id': sample['bin_id'],
                'true_biomass_mg': sample['true_biomass_mg'],
                'predicted_biomass_mg': prediction,
                'absolute_error': abs(prediction - sample['true_biomass_mg']) if prediction is not None else None,
                'relative_error': abs(prediction - sample['true_biomass_mg']) / sample['true_biomass_mg'] if prediction is not None else None
            }
            results.append(result)
        
        # Calculate metrics
        valid_results = [r for r in results if r['predicted_biomass_mg'] is not None]
        
        if valid_results:
            mae = np.mean([r['absolute_error'] for r in valid_results])
            mape = np.mean([r['relative_error'] for r in valid_results]) * 100
            
            print(f"\n📊 Validation Results:")
            print(f"   - Valid predictions: {len(valid_results)}/{len(results)}")
            print(f"   - Mean Absolute Error: {mae:.2f} mg")
            print(f"   - Mean Absolute Percentage Error: {mape:.1f}%")
            
            # Show sample predictions
            print(f"\n📋 Sample Predictions:")
            for i, result in enumerate(valid_results[:10]):
                print(f"   {i+1}. {result['specimen_id']}: {result['true_biomass_mg']:.1f} mg → {result['predicted_biomass_mg']:.1f} mg (error: {result['absolute_error']:.1f} mg)")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict biomass from specimen images')
    parser.add_argument('--model', default='optimized_model_outputs/optimized_biomass_model_20250716_201424.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--test', action='store_true', help='Test on validation set')
    parser.add_argument('--bin_mapping', default='bin_results/bin_id_biomass_mapping.csv',
                       help='Path to BIN mapping CSV')
    parser.add_argument('--image_index', default='image_index.json',
                       help='Path to image index JSON')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = BiomassPredictor(args.model)
    
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"🔄 Predicting biomass for {args.image}...")
        prediction = predictor.predict_single_image(args.image)
        
        if prediction is not None:
            print(f"✅ Predicted biomass: {prediction:.2f} mg")
        else:
            print(f"❌ Failed to predict biomass")
    
    elif args.test:
        # Test on validation set
        predictor.test_on_validation_set(
            args.bin_mapping, 
            args.image_index, 
            args.num_samples
        )
    
    else:
        print("Please specify --image for single prediction or --test for validation")

if __name__ == "__main__":
    main()