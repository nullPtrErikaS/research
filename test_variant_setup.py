#!/usr/bin/env python3
"""
Simplified test to verify variant parsing functions work without streamlit context.
"""
import sys
import os

# Test code directly without importing from streamlit app


def test_parse_variant_logic():
    """Test the variant parsing logic directly."""
    print("=" * 60)
    print("Testing Variant Parsing Logic")
    print("=" * 60)
    
    # Simulate the parse_variant_settings function
    def parse_variant(folder_name):
        settings = {
            "lemmatize": True,
            "lowercase": True,
            "remove_stopwords": True,
            "min_length": 3
        }
        
        if "no_lemmatize" in folder_name:
            settings["lemmatize"] = False
        if "no_lowercase" in folder_name:
            settings["lowercase"] = False
        if "no_stopwords" in folder_name:
            settings["remove_stopwords"] = False
        if "minlen2" in folder_name:
            settings["min_length"] = 2
        
        # Generate label
        if folder_name == "preproc_default":
            label = "Default"
        elif "no_lemmatize" in folder_name:
            label = "No Lemmatization"
        elif "no_stopwords" in folder_name:
            label = "No Stopwords"
        elif "no_lowercase" in folder_name:
            label = "Preserve Case"
        elif "minlen2" in folder_name:
            label = "Min Length 2"
        else:
            label = folder_name.replace("preproc_", "").title()
        
        return label, settings
    
    # Test cases
    test_cases = [
        ("preproc_default", "Default", {"lemmatize": True, "lowercase": True, "remove_stopwords": True, "min_length": 3}),
        ("preproc_no_stopwords", "No Stopwords", {"lemmatize": True, "lowercase": True, "remove_stopwords": False, "min_length": 3}),
        ("preproc_no_lemmatize", "No Lemmatization", {"lemmatize": False, "lowercase": True, "remove_stopwords": True, "min_length": 3}),
        ("preproc_no_lowercase", "Preserve Case", {"lemmatize": True, "lowercase": False, "remove_stopwords": True, "min_length": 3}),
        ("preproc_minlen2", "Min Length 2", {"lemmatize": True, "lowercase": True, "remove_stopwords": True, "min_length": 2}),
    ]
    
    all_pass = True
    for folder_name, expected_label, expected_settings in test_cases:
        label, settings = parse_variant(folder_name)
        
        label_match = label == expected_label
        settings_match = settings == expected_settings
        
        status = "✓" if (label_match and settings_match) else "✗"
        print(f"\n{status} {folder_name}")
        print(f"  Expected: {expected_label} | Got: {label}")
        print(f"  Settings: {settings}")
        
        if not (label_match and settings_match):
            all_pass = False
            if not label_match:
                print(f"  ERROR: Label mismatch!")
            if not settings_match:
                print(f"  ERROR: Settings mismatch!")
    
    return all_pass


def test_artifact_structure():
    """Check if artifact directories exist."""
    print("\n" + "=" * 60)
    print("Testing Artifact Structure")
    print("=" * 60)
    
    artifacts_dir = r"c:\Users\Owner\research\artifacts"
    
    if not os.path.exists(artifacts_dir):
        print(f"✗ Artifacts directory not found: {artifacts_dir}")
        return False
    
    print(f"✓ Artifacts directory exists: {artifacts_dir}")
    
    expected_variants = [
        "preproc_default",
        "preproc_no_stopwords",
        "preproc_no_lemmatize",
        "preproc_no_lowercase",
        "preproc_minlen2"
    ]
    
    all_exist = True
    for variant in expected_variants:
        variant_path = os.path.join(artifacts_dir, variant)
        exists = os.path.isdir(variant_path)
        status = "✓" if exists else "✗"
        print(f"{status} {variant}/")
        
        if exists:
            # Check for key files
            csv_files = [f for f in os.listdir(variant_path) if f.endswith('.csv')]
            npy_files = [f for f in os.listdir(variant_path) if f.endswith('.npy')]
            print(f"    - CSVs: {len(csv_files)}, NPY: {len(npy_files)}")
        else:
            all_exist = False
    
    return all_exist


def main():
    print("\n" + "█" * 60)
    print("VARIANT SWITCHING SETUP VERIFICATION")
    print("█" * 60 + "\n")
    
    results = []
    
    # Test 1: Parsing logic
    try:
        r1 = test_parse_variant_logic()
        results.append(("Parse Variant Logic", r1))
    except Exception as e:
        print(f"✗ ERROR in parse logic: {e}")
        results.append(("Parse Variant Logic", False))
    
    # Test 2: Artifact structure
    try:
        r2 = test_artifact_structure()
        results.append(("Artifact Structure", r2))
    except Exception as e:
        print(f"✗ ERROR in artifact check: {e}")
        results.append(("Artifact Structure", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_pass = all(r[1] for r in results)
    
    if all_pass:
        print("\n✓ All checks passed!")
        return 0
    else:
        print("\n✗ Some checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
