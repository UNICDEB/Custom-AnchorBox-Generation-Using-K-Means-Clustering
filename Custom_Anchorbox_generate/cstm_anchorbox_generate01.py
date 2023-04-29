import numpy as np
from sklearn.cluster import KMeans

def generate_anchors(annotations, num_clusters=9):
    """Generates anchor boxes using KMeans clustering in a custom dataset.
    
    Args:
        annotations: List of bounding box annotations in the form of [width, height].
        num_clusters: Number of anchor boxes to generate.
        
    Returns:
        anchor_boxes: List of anchor boxes in the form of [width, height].
    """
    # Convert annotations to a numpy array
    annotations = np.array(annotations)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(annotations)
    anchor_boxes = kmeans.cluster_centers_
    
    return anchor_boxes
