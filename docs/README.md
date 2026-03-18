# Embeddings Explorer Documentation

Welcome to the Embeddings Explorer! This interactive application helps you map, analyze, and explore the relationships within your short-text dataset using advanced natural language processing (NLP) and dimensionality reduction (PCA, t-SNE, UMAP).

This document serves as the official guide to using the application, outlining its primary features and common analytical workflows.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Selection & Grouping](#selection--grouping)
4. [Document Analysis and Comparison](#document-analysis-and-comparison)
5. [Visual Settings](#visual-settings)
6. [Analytical Workflows](#analytical-workflows)

---

## Getting Started

1. **Launch the Application**: Run `streamlit run streamlit_app/streamlit_app.py` from your terminal. 
2. **Data Loading**: By default, the application will automatically scan your `artifacts/` directories to mount the most recent preprocessed dataset (e.g., `processed_data_with_clusters.csv`) and projection files. No manual loading is required!
3. **Navigating the App**: The left sidebar contains all of your global tools (Coloring, Visual Settings, Saved Groups, Filtering), while the main console contains your interactive projections and analytical dashboards.

---

## Interface Overview

### The 3 Projections
The tool visualizes your documents in three different two-dimensional algorithms simultaneously:
* **t-SNE** (Top Left): Excellent for grouping distinct clusters and identifying tight, local relationships.
* **UMAP** (Top Right): Balances local grouping with global structure. Good for seeing how different clusters relate to each other.
* **PCA** (Bottom Left): A linear projection. Excellent for seeing absolute variations and the "overall spread" or variance in the raw data.

### Interaction Models
* **Hovering**: Hover over any point to read a quick snippet of the document and see its ID and core cluster.
* **Clicking**: Click a single point to select it.
* **Lassoing**: Click and drag your mouse to lasso a group of points.

---

## Selection & Grouping

Selections are a first-class citizen in the Explorer. Selecting points in any of the three plots syncs the selection across all of them instantly.

### The Selection Control Bar
Whenever you select one or more documents, a persistent **Control Bar** appears at the top of the "Explore" canvas, displaying the exact count of your current selection. 
* **Clear Selection**: Instantly drops your target and resets the canvas.
* **Undo**: Made a selection mistake? The explorer tracks your last 10 selections. Click "Undo" to naturally step backwards in time.
* **Selection History**: Click this dropdown to jump directly back to a previous selection state.
* **Save Group**: If you are satisfied with a specific selection, click this to store it as a **Saved Group** (previously known as a Cohort). 

### Saved Groups (Sidebar)
In the left sidebar, the **Saved Groups** expander manages selections you want to keep long-term.
* Name your groups descriptively (e.g. "Outliers", "Color Keyword Documents").
* Saved groups can be toggled on/off to instantly re-select those exact documents on the graph.
* **Color Mode**: Change "Color points by:" in the sidebar to **"Saved Group"** to visually distinct your saved groups from the rest of the dataset with custom coloring.

---

## Document Analysis and Comparison

The "Analysis" section (below the graphs) dynamically changes its layout depending on how many documents you have selected.

### Multi-Document View (3+ Selected)
Selecting a wide swath of documents allows you to inspect bulk trends:
* **Selected Documents Table**: Allows you to read the full text of all selected documents.
* **Cluster Stats**: A bar chart mapping how many documents from each distinct semantic cluster are present in your selection.
* **Top Keywords**: A frequency plot revealing the most common vocabulary terms active inside your selected bubble.

### Single Document View (1 Selected)
Selecting a single document puts the app in "Focus" mode:
* **Deep Reading**: Produces a rich, full-card view of the document's metadata.
* **Nearest Neighbors**: Calculates and displays the top 5 most semantically similar documents in the entire dataset using high-dimensional Cosine Similarity. 
* *Note: When focused on a single document, the graph collapses to a single large projection for detailed inspection. Change back by clearing your selection.*

### Two-Document Comparison (2 Selected)
Selecting *exactly* two documents unlocks special comparative analysis, perfect for validating "Opposites" or examining specific relationships:
* **Quantitative Distances**: Displays the absolute Cosine Similarity, Euclidean Distance, and the literal visual distances between the points in all three projection plots.
* **Shared Terms**: Extracts and lists up to 15 vocabulary words the two documents explicitly share.
* **The "Midpoint Analyzer"**: This tool mathematically calculates the exact vector midpoint between your two chosen documents. It then queries the nearest neighbors to that midpoint, returning the **8 documents that sit conceptually "between" your selection**, bridging their semantic gap.

---

## Visual Settings

In the left sidebar under "Visual Tweaks", you can control the graphics engine.

* **Fading & Focus Rings**: To maximize visibility, unselected background points are aggressively faded into the background (0.05 opacity) while your selected points remain sharp. Selecting 3 or more points will also generate a **Focus Ring**—a dashed red convex hull enclosing your selected space on the scatterplot. 
* **Point Alpha (Opacity)**: Manually slider to adjust baseline opacity.
* **Point Size**: Slide to increase or decrease the visual density.

---

## Analytical Workflows

### 1. Validating "Opposite" Documents
If you believe two documents are conceptual opposites:
1. Search for their IDs using the **"Search & Filter"** tool in the sidebar.
2. Select both of them on the graph.
3. Validate using the **Two-Document Comparison** view. Check if their Cosine Similarity is near 0.0. Read the "Shared terms". Then review the "Midpoint" table to identify the texts that bridge the transition between your two extremes.

### 2. Investigating Keyword Trends
If you want to understand how a specific topic is treated:
1. Open the **"Search & Filter"** tool and type your keyword (e.g., "color"). Ensure the "Phrase/Text" scope is checked.
2. Your hits will permanently turn bright Orange.
3. Lasso all the Orange hits. 
4. Check the **"Top Keywords"** analysis panel to see what *other* words trend commonly alongside your query.
5. Save this lasso selection as a **"Saved Group"** named for your query to retain it for later comparison.
