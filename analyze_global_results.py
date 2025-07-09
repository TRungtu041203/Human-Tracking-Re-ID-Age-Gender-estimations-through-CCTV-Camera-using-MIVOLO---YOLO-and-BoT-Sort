#!/usr/bin/env python3
"""
Example script to analyze global temporal smoothing results.
Usage: python analyze_global_results.py global_smoothing_results.json
"""

import json
import sys
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


def load_results(file_path: str) -> Dict[str, Any]:
    """Load global smoothing results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_summary(results: Dict[str, Any]):
    """Print a summary of the results."""
    final_predictions = results["final_predictions"]
    track_stats = results["track_statistics"]
    
    print(f"\n{'='*60}")
    print(f"GLOBAL TEMPORAL SMOOTHING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total tracked individuals: {len(final_predictions)}")
    
    # Age statistics
    ages = [pred["age"] for pred in final_predictions.values()]
    print(f"\nAge Statistics:")
    print(f"  Mean age: {np.mean(ages):.1f} years")
    print(f"  Age range: {np.min(ages):.1f} - {np.max(ages):.1f} years")
    
    # Gender distribution
    genders = [pred["gender"] for pred in final_predictions.values()]
    gender_counts = {g: genders.count(g) for g in set(genders)}
    print(f"\nGender Distribution:")
    for gender, count in gender_counts.items():
        print(f"  {gender.capitalize()}: {count} ({count/len(genders)*100:.1f}%)")
    
    # Detection quality
    detections = [pred["num_detections"] for pred in final_predictions.values()]
    print(f"\nDetection Quality:")
    print(f"  Average detections per track: {np.mean(detections):.1f}")
    print(f"  Detection range: {np.min(detections)} - {np.max(detections)} frames")


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed results for each track."""
    final_predictions = results["final_predictions"]
    track_stats = results["track_statistics"]
    
    print(f"\n{'='*80}")
    print(f"DETAILED TRACK RESULTS")
    print(f"{'='*80}")
    print(f"{'Track ID':<10} {'Final Age':<12} {'Final Gender':<14} {'Detections':<12} {'Age Std':<10} {'Confidence':<12}")
    print(f"{'-'*80}")
    
    for track_id in sorted(final_predictions.keys(), key=int):
        pred = final_predictions[track_id]
        stats = track_stats[track_id]
        
        print(f"{track_id:<10} {pred['age']:<12.1f} {pred['gender']:<14} "
              f"{pred['num_detections']:<12} {stats['age_std']:<10.2f} "
              f"{stats['gender_confidence']:<12.3f}")


def create_visualizations(results: Dict[str, Any], save_plots: bool = True):
    """Create visualizations of the results."""
    final_predictions = results["final_predictions"]
    track_stats = results["track_statistics"]
    
    # Extract data
    track_ids = list(final_predictions.keys())
    ages = [final_predictions[tid]["age"] for tid in track_ids]
    genders = [final_predictions[tid]["gender"] for tid in track_ids]
    detections = [final_predictions[tid]["num_detections"] for tid in track_ids]
    age_stds = [track_stats[tid]["age_std"] for tid in track_ids]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Global Temporal Smoothing Results Analysis', fontsize=16)
    
    # Age distribution
    ax1.hist(ages, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Number of Tracks')
    ax1.set_title('Age Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Gender distribution
    gender_counts = {g: genders.count(g) for g in set(genders)}
    colors = ['lightcoral', 'lightblue']
    ax2.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax2.set_title('Gender Distribution')
    
    # Detection quality vs Age stability
    scatter = ax3.scatter(detections, age_stds, c=ages, cmap='viridis', alpha=0.7, s=50)
    ax3.set_xlabel('Number of Detections')
    ax3.set_ylabel('Age Standard Deviation')
    ax3.set_title('Detection Quality vs Age Stability')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Final Age')
    
    # Age vs Detection count
    colors_gender = ['red' if g == 'male' else 'blue' for g in genders]
    ax4.scatter(ages, detections, c=colors_gender, alpha=0.7, s=50)
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Number of Detections')
    ax4.set_title('Age vs Detection Count (Red=Male, Blue=Female)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('global_smoothing_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'global_smoothing_analysis.png'")
    
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_global_results.py <global_smoothing_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    try:
        results = load_results(results_file)
        print_summary(results)
        print_detailed_results(results)
        
        # Ask user if they want to create visualizations
        try:
            create_viz = input("\nCreate visualizations? (y/n): ").lower().strip()
            if create_viz in ['y', 'yes']:
                create_visualizations(results)
        except ImportError:
            print("\nMatplotlib not available. Skipping visualizations.")
            print("Install with: pip install matplotlib")
        
    except FileNotFoundError:
        print(f"Error: File '{results_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{results_file}'.")
        sys.exit(1)


if __name__ == "__main__":
    main() 