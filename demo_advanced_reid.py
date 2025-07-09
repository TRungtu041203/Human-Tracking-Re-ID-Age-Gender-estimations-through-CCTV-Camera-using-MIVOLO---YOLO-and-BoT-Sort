#!/usr/bin/env python3
"""
Advanced ReID Model Demo for MiVOLO
Demonstrates the use of state-of-the-art Re-ID backbones for improved tracking accuracy.
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys

def run_command(cmd, description):
    """Run a command with status reporting."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def setup_models():
    """Setup and download Re-ID models."""
    print("🚀 Setting up Advanced Re-ID Models for BoT-SORT")
    print("=" * 60)
    
    # Check if setup script exists
    if not Path("setup_reid_models.py").exists():
        print("❌ setup_reid_models.py not found!")
        return False
    
    # List available models
    if not run_command([sys.executable, "setup_reid_models.py", "--list"], 
                      "Listing available Re-ID models"):
        return False
    
    # Get recommendation for balanced use case
    if not run_command([sys.executable, "setup_reid_models.py", "--recommend", "balanced"], 
                      "Getting model recommendation"):
        return False
    
    return True

def download_and_configure_model(model_name="osnet_x1_0_msmt17.pt"):
    """Download and configure a specific Re-ID model."""
    print(f"\n📥 Downloading and configuring {model_name}")
    
    # Download the model
    if not run_command([sys.executable, "setup_reid_models.py", "--download", model_name], 
                      f"Downloading {model_name}"):
        return False
    
    # Configure botsort.yaml to use the model
    if not run_command([sys.executable, "setup_reid_models.py", "--use", model_name], 
                      f"Configuring BoT-SORT to use {model_name}"):
        return False
    
    return True

def run_tracking_demo(input_source, model_name="osnet_x1_0_msmt17.pt"):
    """Run MiVOLO tracking with advanced Re-ID model."""
    print(f"\n🎬 Running tracking demo with {model_name}")
    
    # Check if required model files exist
    detector_weights = "models/yolov8x_person_face.pt"
    checkpoint = "models/mivolo_imbd.pth.tar"
    
    if not Path(detector_weights).exists():
        print(f"❌ Detector weights not found: {detector_weights}")
        print("Please download the detector model first.")
        return False
    
    if not Path(checkpoint).exists():
        print(f"❌ MiVOLO checkpoint not found: {checkpoint}")
        print("Please download the MiVOLO model first.")
        return False
    
    # Prepare output directory
    output_dir = f"output/advanced_reid_{model_name.replace('.pt', '')}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run tracking with confidence-based selection and advanced ReID
    cmd = [
        sys.executable, "demo.py",
        "--input", input_source,
        "--output", output_dir,
        "--detector-weights", detector_weights,
        "--checkpoint", checkpoint,
        "--tracker", "botsort.yaml",
        "--confidence-based",
        "--min-detection-conf", "0.5",
        "--min-gender-conf", "0.6",
        "--with-persons",
        "--draw"
    ]
    
    if not run_command(cmd, f"Running MiVOLO tracking with {model_name}"):
        return False
    
    print(f"✅ Tracking completed! Results saved to: {output_dir}")
    return True

def compare_models(input_source):
    """Compare tracking performance with different Re-ID models."""
    print("\n🔬 Comparing Re-ID Model Performance")
    print("=" * 50)
    
    models_to_test = [
        ("osnet_x0_5_msmt17.pt", "Fast OSNet"),
        ("osnet_x1_0_msmt17.pt", "Balanced OSNet"), 
        ("clip_vit_b16_market1501.pt", "CLIP ViT-B"),
    ]
    
    results = {}
    
    for model, description in models_to_test:
        print(f"\n🧪 Testing {description} ({model})")
        
        # Download and configure model
        if not download_and_configure_model(model):
            print(f"❌ Failed to setup {model}")
            continue
        
        # Run tracking and measure time
        start_time = time.time()
        success = run_tracking_demo(input_source, model)
        end_time = time.time()
        
        if success:
            processing_time = end_time - start_time
            results[model] = {
                "description": description,
                "time": processing_time,
                "success": True
            }
            print(f"✅ {description} completed in {processing_time:.2f} seconds")
        else:
            results[model] = {
                "description": description,
                "success": False
            }
            print(f"❌ {description} failed")
    
    # Print comparison results
    print("\n📊 Performance Comparison Results:")
    print("-" * 50)
    for model, result in results.items():
        if result["success"]:
            print(f"{result['description']:20} | {result['time']:8.2f}s | {model}")
        else:
            print(f"{result['description']:20} | {'FAILED':8} | {model}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Advanced ReID Model Demo for MiVOLO")
    parser.add_argument("--input", type=str, default="https://www.youtube.com/shorts/pVh32k0hGEI",
                       help="Input video source (file, URL, or webcam)")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup models without running demo")
    parser.add_argument("--model", type=str, default="osnet_x1_0_msmt17.pt",
                       help="Specific Re-ID model to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple Re-ID models")
    parser.add_argument("--quick-demo", action="store_true",
                       help="Run quick demo with recommended model")
    
    args = parser.parse_args()
    
    print("🎯 Advanced Re-ID Model Demo for MiVOLO Tracking")
    print("=" * 60)
    print("This demo showcases state-of-the-art Re-ID backbones for improved tracking accuracy")
    print()
    
    # Setup models
    if not setup_models():
        print("❌ Failed to setup Re-ID models")
        return 1
    
    if args.setup_only:
        print("✅ Model setup completed!")
        return 0
    
    if args.compare:
        # Compare multiple models
        results = compare_models(args.input)
        
        # Recommend best model based on results
        successful_models = {k: v for k, v in results.items() if v["success"]}
        if successful_models:
            fastest_model = min(successful_models.items(), key=lambda x: x[1]["time"])
            print(f"\n💡 Fastest model: {fastest_model[1]['description']} ({fastest_model[0]})")
        
    elif args.quick_demo:
        # Quick demo with recommended model
        print("🚀 Running quick demo with recommended OSNet model")
        if download_and_configure_model("osnet_x1_0_msmt17.pt"):
            run_tracking_demo(args.input, "osnet_x1_0_msmt17.pt")
        
    else:
        # Run demo with specific model
        if download_and_configure_model(args.model):
            run_tracking_demo(args.input, args.model)
    
    print("\n🎉 Demo completed!")
    print("\n📋 Next Steps:")
    print("1. Check the output directory for tracking results")
    print("2. Analyze confidence_based_results.json for detailed metrics")
    print("3. Try different Re-ID models for your specific use case")
    print("4. Tune parameters in botsort.yaml for optimal performance")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1) 