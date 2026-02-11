
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock streamlit before importing the app
import sys
from unittest.mock import MagicMock

# Create a mock streamlit module
mock_st = MagicMock()
mock_st.session_state = {}
mock_st.sidebar = MagicMock()
mock_st.expander = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
mock_st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
mock_st.cache_data = lambda func: func
sys.modules['streamlit'] = mock_st
sys.modules['streamlit.components.v1'] = MagicMock()
sys.modules['streamlit_plotly_events'] = MagicMock()

# Since we can't easily import the script as a module without running it, 
# we will simulate the logic we just implemented.

class TestSelectionLogic(unittest.TestCase):
    def setUp(self):
        # Reset session state
        mock_st.session_state = {
            'selected_ids': [],
            'selection_history': [],
            'last_chart_states': {},
            'additive_mode': False,
            'chunk_parent_map': {},
            'parent_chunk_map': {},
            'dragmode': 'lasso'
        }
        
        # Mock Dataframe
        self.df = pd.DataFrame({
            'doc_id': ['doc1', 'doc2', 'doc3', 'doc4'],
            'val': [1, 2, 3, 4]
        })
        
    def expand_chunk_links(self, ids):
        # simplified version of the function in the app
        return list(set(ids))

    def update_selection(self, doc_ids, additive=False):
        # Logic from the app
        current = list(mock_st.session_state.get('selected_ids', []))
        if additive:
            base_seq = current + list(doc_ids)
        else:
            base_seq = list(doc_ids)
        mock_st.session_state['selected_ids'] = list(set(base_seq))

    def run_selection_logic(self):
        # The logic we inserted into the app
        chart_keys = ['chart_tsne', 'chart_umap', 'chart_pca']
        for key in chart_keys:
            current_state = mock_st.session_state.get(key)
            last_state = mock_st.session_state['last_chart_states'].get(key)
            
            if current_state != last_state:
                mock_st.session_state['last_chart_states'][key] = current_state
                
                new_indices = []
                if current_state:
                    if hasattr(current_state, 'selection'):
                        new_indices = current_state.selection.get('point_indices', [])
                    elif isinstance(current_state, dict):
                        new_indices = current_state.get('selection', {}).get('point_indices', [])
                
                selected_docs = []
                if new_indices:
                    # Filter out out-of-bounds indices
                    valid_indices = [i for i in new_indices if 0 <= i < len(self.df)]
                    selected_docs = self.df.iloc[valid_indices]['doc_id'].tolist()
                    
                self.update_selection(selected_docs, additive=mock_st.session_state.get('additive_mode', False))
                break

    def test_new_selection(self):
        # Simulate a selection in TSNE
        selection_obj = MagicMock()
        selection_obj.selection = {'point_indices': [0, 2]}
        
        mock_st.session_state['chart_tsne'] = selection_obj
        
        self.run_selection_logic()
        
        self.assertEqual(sorted(mock_st.session_state['selected_ids']), ['doc1', 'doc3'])
        self.assertEqual(mock_st.session_state['last_chart_states']['chart_tsne'], selection_obj)

    def test_selection_update_pca(self):
        # Simulate a selection in PCA (updating from previous)
        mock_st.session_state['selected_ids'] = ['doc1']
        
        selection_obj = MagicMock()
        selection_obj.selection = {'point_indices': [1]} # doc2
        
        mock_st.session_state['chart_pca'] = selection_obj
        
        self.run_selection_logic()
        
        # Should replace
        self.assertEqual(mock_st.session_state['selected_ids'], ['doc2'])

    def test_additive_selection(self):
        mock_st.session_state['selected_ids'] = ['doc1']
        mock_st.session_state['additive_mode'] = True
        
        selection_obj = MagicMock()
        selection_obj.selection = {'point_indices': [1]} # doc2
        
        mock_st.session_state['chart_umap'] = selection_obj
        
        self.run_selection_logic()
        
        self.assertEqual(sorted(mock_st.session_state['selected_ids']), ['doc1', 'doc2'])

    def test_deselection(self):
        mock_st.session_state['selected_ids'] = ['doc1', 'doc2']
        
        # Simulate empty selection
        selection_obj = MagicMock()
        selection_obj.selection = {'point_indices': []}
        
        mock_st.session_state['chart_tsne'] = selection_obj
        
        self.run_selection_logic()
        
        self.assertEqual(mock_st.session_state['selected_ids'], [])

    def test_no_change(self):
        # state is same as last time
        selection_obj = MagicMock()
        selection_obj.selection = {'point_indices': [0]}
        
        mock_st.session_state['chart_tsne'] = selection_obj
        mock_st.session_state['last_chart_states']['chart_tsne'] = selection_obj
        mock_st.session_state['selected_ids'] = ['doc99'] # existing selection unrelated to chart
        
        self.run_selection_logic()
        
        # Should NOT update because state didn't change
        self.assertEqual(mock_st.session_state['selected_ids'], ['doc99'])

if __name__ == '__main__':
    unittest.main()
