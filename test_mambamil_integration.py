#!/usr/bin/env python3
"""
Test script to verify MambaMIL integration with PIANO library
"""

import sys
import os
import torch

# Add PIANO to path
sys.path.append('/mnt/sdb/ljw/PIANO-Update/PIANO_Preview/PIANO')

def test_mambamil_integration():
    """Test MambaMIL integration with create_mil_model function"""
    
    try:
        from piano.model.mil_factory import create_mil_model, get_mil_model_names, get_mil_default_params
        
        print("✓ Successfully imported PIANO MIL factory functions")
        
        # Check if mambamil is in available models
        available_models = get_mil_model_names()
        print(f"Available MIL models: {available_models}")
        
        if 'mambamil' in available_models:
            print("✓ MambaMIL is registered in available models")
        else:
            print("✗ MambaMIL is NOT registered in available models")
            return False
        
        # Check default parameters
        default_params = get_mil_default_params('mambamil')
        print(f"MambaMIL default parameters: {default_params}")
        
        expected_keys = ['dim_in', 'num_classes', 'dropout', 'act', 'survival', 'layer', 'rate', 'type']
        for key in expected_keys:
            if key in default_params:
                print(f"✓ Parameter '{key}' found in default params: {default_params[key]}")
            else:
                print(f"✗ Parameter '{key}' NOT found in default params")
                return False
        
        # Test model creation with default parameters
        print("\nTesting model creation with default parameters...")
        try:
            model = create_mil_model('mambamil')
            print(f"✓ Successfully created MambaMIL model: {type(model).__name__}")
            print(f"  - Model type: {model.type}")
            print(f"  - Number of classes: {model.num_classes}")
            print(f"  - Number of layers: {len(model.layers)}")
            
            # Test forward pass with new input/output format
            print("\nTesting forward pass with new input/output format...")
            
            # Test with dictionary input
            batch_size, num_patches, feature_dim = 1, 100, 1024
            dummy_features = torch.randn(batch_size, num_patches, feature_dim)
            dummy_labels = torch.tensor([1])
            
            input_dict = {
                'features': dummy_features,
                'labels': dummy_labels
            }
            
            with torch.no_grad():
                output = model(input_dict, return_loss=True)
                
            # Check output format
            required_keys = ['logits', 'raw_attn', 'features', 'loss']
            for key in required_keys:
                if key in output:
                    print(f"  ✓ Output contains '{key}': {output[key].shape if hasattr(output[key], 'shape') else type(output[key])}")
                else:
                    print(f"  ✗ Output missing '{key}'")
                    return False
            
            # Test backward compatibility (direct tensor input)
            print("\nTesting backward compatibility (direct tensor input)...")
            with torch.no_grad():
                output_compat = model(dummy_features, return_loss=False)
                
            if 'logits' in output_compat and output_compat['loss'] is None:
                print("  ✓ Backward compatibility works correctly")
            else:
                print("  ✗ Backward compatibility failed")
                return False
                
        except Exception as e:
            print(f"✗ Failed to create or test MambaMIL model with default parameters: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test model creation with custom parameters
        print("\nTesting model creation with custom parameters...")
        try:
            custom_params = {
                'dim_in': 2048,
                'num_classes': 5,
                'dropout': 0.3,
                'act': 'relu',
                'layer': 3,
                'type': 'Mamba'
            }
            model = create_mil_model('mambamil', **custom_params)
            print(f"✓ Successfully created MambaMIL model with custom parameters")
            print(f"  - Model type: {model.type}")
            print(f"  - Number of classes: {model.num_classes}")
            print(f"  - Number of layers: {len(model.layers)}")
            
            # Test with custom input dimensions
            batch_size, num_patches, feature_dim = 1, 50, 2048
            dummy_features = torch.randn(batch_size, num_patches, feature_dim)
            dummy_labels = torch.tensor([3])  # Class 3 out of 5
            
            input_dict = {
                'features': dummy_features,
                'labels': dummy_labels
            }
            
            with torch.no_grad():
                output = model(input_dict, return_loss=True)
                
            print(f"  ✓ Custom model forward pass successful")
            print(f"    - Logits shape: {output['logits'].shape}")
            print(f"    - Features shape: {output['features'].shape}")
            print(f"    - Loss: {output['loss'].item() if output['loss'] is not None else 'None'}")
            
        except Exception as e:
            print(f"✗ Failed to create MambaMIL model with custom parameters: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 All tests passed! MambaMIL is successfully integrated into PIANO library.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_mambamil_integration()
    sys.exit(0 if success else 1)
