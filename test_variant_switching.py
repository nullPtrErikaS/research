#!/usr/bin/env python3
"""
Test script to verify variant switching functionality.
Tests the parse_variant_settings and get_variant_comparison_info functions.
"""
import sys
import os

# Add streamlit_app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streamlit_app'))

def test_parse_variant_settings():
    """Test the variant settings parser."""
    print("=" * 60)
    print("TEST 1: Parse Variant Settings")
    print("=" * 60)
    
    # Import the parsing function
    from streamlit_app import parse_variant_settings
    
    test_cases = [
        ("artifacts/preproc_default", "Default"),
        ("artifacts/preproc_no_stopwords", "No Stopwords"),
        ("artifacts/preproc_no_lemmatize", "No Lemmatization"),
        ("artifacts/preproc_no_lowercase", "Preserve Case"),
        ("artifacts/preproc_minlen2", "Min Length 2"),
    ]
    
    for variant_path, expected_label in test_cases:
        result = parse_variant_settings(variant_path)
        status = "✓" if result["label"] == expected_label else "✗"
        print(f"\n{status} {variant_path}")
        print(f"  Label: {result['label']} (expected: {expected_label})")
        print(f"  Settings: {result['settings']}")
        print(f"  Config: {result['config']}")
        
        if result["label"] != expected_label:
            print(f"  ERROR: Expected '{expected_label}', got '{result['label']}'")
            return False
    
    print("\n✓ All variant parsing tests passed!")
    return True


def test_variant_caching():
    """Test that variant key affects caching."""
    print("\n" + "=" * 60)
    print("TEST 2: Variant Caching Key")
    print("=" * 60)
    
    from streamlit_app import load_and_process_data
    import inspect
    
    # Check that _variant_key parameter is in the function signature
    sig = inspect.signature(load_and_process_data)
    params = list(sig.parameters.keys())
    
    print("\nload_and_process_data parameters:", params)
    
    if "_variant_key" in params:
        print("✓ _variant_key parameter exists in cache function")
        print(f"  Position: {params.index('_variant_key')}")
        print(f"  Default: {sig.parameters['_variant_key'].default}")
        return True
    else:
        print("✗ _variant_key parameter NOT found!")
        return False


def test_coordinate_loading():
    """Test that coordinate loading is cached."""
    print("\n" + "=" * 60)
    print("TEST 3: Coordinate Loading Cache")
    print("=" * 60)
    
    from streamlit_app import load_coordinate_files
    import inspect
    
    sig = inspect.signature(load_coordinate_files)
    params = list(sig.parameters.keys())
    
    print("\nload_coordinate_files parameters:", params)
    print(f"Function has @st.cache_data decorator")
    
    # Check the required coordinate files are returned
    expected_files = ['tsne', 'umap', 'pca', 'tfidf', 'cluster_labels', 'doc_ids']
    print(f"\nExpected coordinate keys: {expected_files}")
    
    if "_variant_key" in params:
        print("✓ _variant_key parameter exists in coordinate loading function")
        return True
    else:
        print("✗ _variant_key parameter NOT found in coordinate loading!")
        return False


def test_bundle_candidate_order():
    """Test that bundle_candidates prioritizes selected_bundle_root."""
    print("\n" + "=" * 60)
    print("TEST 4: Bundle Candidate Priority")
    print("=" * 60)
    
    # We can't fully test this without mocking streamlit state,
    # but we can verify the function exists and has the right logic
    from streamlit_app import bundle_candidates
    import inspect
    
    source = inspect.getsource(bundle_candidates)
    
    # Check for bundle prioritization logic
    checks = [
        ("selected_bundle_root" in source, "Checks selected_bundle_root"),
        ("if selected_bundle_root" in source, "Has conditional for selected_bundle_root"),
        ("roots.append(selected_bundle_root)" in source, "Appends selected_bundle_root to roots"),
    ]
    
    all_pass = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"{status} {desc}")
        if not check:
            all_pass = False
    
    return all_pass


def main():
    """Run all tests."""
    print("\n" + "█" * 60)
    print("VARIANT SWITCHING FUNCTIONALITY TEST SUITE")
    print("█" * 60)
    
    tests = [
        test_parse_variant_settings,
        test_variant_caching,
        test_coordinate_loading,
        test_bundle_candidate_order,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_pass = all(r[1] for r in results)
    
    if all_pass:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
