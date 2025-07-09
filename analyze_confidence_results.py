#!/usr/bin/env python3
"""
Analyze confidence-based selection results from MiVOLO.
Usage: python analyze_confidence_results.py confidence_based_results.json
"""

import json
import sys
import numpy as np
from typing import Dict, Any


def print_summary(results: Dict[str, Any]):
    """Print a summary of the confidence-based results."""
    confidence_results = results["confidence_selection_results"]
    stats = results["statistics"]
    
    print(f"\n{'='*60}")
    print(f"CONFIDENCE-BASED SELECTION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total tracked individuals: {len(confidence_results)}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Average detection confidence: {stats['avg_detection_conf']:.3f}")
    print(f"  Average gender confidence: {stats['avg_gender_conf']:.3f}")
    print(f"  Average composite confidence: {stats['avg_composite_conf']:.3f}")
    print(f"  Confidence range: {stats['min_composite_conf']:.3f} - {stats['max_composite_conf']:.3f}")
    
    # Age statistics
    ages = [pred["age"] for pred in confidence_results.values()]
    print(f"\nAge Statistics:")
    print(f"  Mean age: {np.mean(ages):.1f} years")
    print(f"  Age range: {np.min(ages):.1f} - {np.max(ages):.1f} years")
    
    # Gender distribution
    genders = [pred["gender"] for pred in confidence_results.values()]
    gender_counts = {g: genders.count(g) for g in set(genders)}
    print(f"\nGender Distribution:")
    for gender, count in gender_counts.items():
        print(f"  {gender.capitalize()}: {count} ({count/len(genders)*100:.1f}%)")
    
    # Best frame distribution
    best_frames = [pred["best_frame"] for pred in confidence_results.values()]
    print(f"\nBest Frame Distribution:")
    print(f"  Earliest best frame: {np.min(best_frames)}")
    print(f"  Latest best frame: {np.max(best_frames)}")
    print(f"  Average best frame: {np.mean(best_frames):.1f}")


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed results for each track."""
    confidence_results = results["confidence_selection_results"]
    
    print(f"\n{'='*90}")
    print(f"DETAILED CONFIDENCE-BASED RESULTS")
    print(f"{'='*90}")
    print(f"{'Track ID':<10} {'Age':<6} {'Gender':<8} {'Det.Conf':<9} {'Gen.Conf':<9} {'Comp.Conf':<10} {'Best Frame':<12}")
    print(f"{'-'*90}")
    
    for track_id in sorted(confidence_results.keys(), key=int):
        pred = confidence_results[track_id]
        print(f"{track_id:<10} {pred['age']:<6.1f} {pred['gender']:<8} "
              f"{pred['detection_confidence']:<9.3f} {pred['gender_confidence']:<9.3f} "
              f"{pred['composite_confidence']:<10.3f} {pred['best_frame']:<12}")


def print_confidence_insights(results: Dict[str, Any]):
    """Print insights about confidence patterns."""
    confidence_results = results["confidence_selection_results"]
    
    print(f"\n{'='*60}")
    print(f"CONFIDENCE INSIGHTS")
    print(f"{'='*60}")
    
    # High confidence tracks
    high_conf_tracks = [
        (track_id, pred) for track_id, pred in confidence_results.items()
        if pred['composite_confidence'] > 0.8
    ]
    
    print(f"High-confidence tracks (>0.8): {len(high_conf_tracks)}")
    if high_conf_tracks:
        avg_high_conf = np.mean([pred['composite_confidence'] for _, pred in high_conf_tracks])
        print(f"  Average confidence: {avg_high_conf:.3f}")
        
        # Show top 3 highest confidence
        top_conf = sorted(high_conf_tracks, key=lambda x: x[1]['composite_confidence'], reverse=True)[:3]
        print(f"  Top 3 highest confidence:")
        for track_id, pred in top_conf:
            print(f"    Track {track_id}: {pred['composite_confidence']:.3f} "
                  f"(Age: {pred['age']:.1f}, Gender: {pred['gender']}, Frame: {pred['best_frame']})")
    
    # Low confidence tracks
    low_conf_tracks = [
        (track_id, pred) for track_id, pred in confidence_results.items()
        if pred['composite_confidence'] < 0.6
    ]
    
    print(f"\nLow-confidence tracks (<0.6): {len(low_conf_tracks)}")
    if low_conf_tracks:
        avg_low_conf = np.mean([pred['composite_confidence'] for _, pred in low_conf_tracks])
        print(f"  Average confidence: {avg_low_conf:.3f}")
        print(f"  These tracks may need manual review")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_confidence_results.py confidence_based_results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
        
        # Validate the structure
        if "confidence_selection_results" not in results or "statistics" not in results:
            print("Error: Invalid confidence results file format")
            sys.exit(1)
        
        print_summary(results)
        print_detailed_results(results)
        print_confidence_insights(results)
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{input_file}'")
        sys.exit(1)


if __name__ == "__main__":
    main() 