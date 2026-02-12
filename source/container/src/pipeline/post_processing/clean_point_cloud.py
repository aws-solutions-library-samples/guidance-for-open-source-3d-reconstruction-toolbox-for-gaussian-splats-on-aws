#!/usr/bin/env python3
"""
Point Cloud Cleaner - Removes outlier points from PLY files using statistical outlier removal
and cluster-based filtering.

Usage:
    python clean_pointcloud.py <input.ply> [--level {low,medium,high}] [--output <output.ply>]
                                           [--min-cluster <size>] [--no-confirm]
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.spatial import KDTree
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip3 install numpy scipy plyfile")
    sys.exit(1)


LEVELS = {
    "low": {"nb_neighbors": 20, "std_ratio": 2.0},
    "medium": {"nb_neighbors": 50, "std_ratio": 1.0},
    "high": {"nb_neighbors": 100, "std_ratio": 0.5},
}


def print_progress(current, total, prefix="Progress", bar_length=40):
    """Print a progress bar."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)
    sys.stdout.write(f"\r{prefix}: [{bar}] {percent*100:.1f}% ({current:,}/{total:,})")
    sys.stdout.flush()
    if current == total:
        print()


def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Remove statistical outliers from point cloud."""
    print(f"  Building KD-tree for {len(points):,} points...")
    tree = KDTree(points)
    
    print(f"  Querying {nb_neighbors} nearest neighbors...")
    n_points = len(points)
    batch_size = 10000
    all_distances = []
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        distances, _ = tree.query(points[i:end], k=nb_neighbors + 1)
        all_distances.append(distances)
        print_progress(end, n_points, "  Analyzing neighbors")
    
    distances = np.vstack(all_distances)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + std_ratio * global_std
    
    return mean_distances <= threshold


def cluster_filter_fast(points: np.ndarray, min_cluster_size: int, eps: float = None):
    """
    Remove small isolated clusters. Returns mask and cluster info for preview.
    """
    if len(points) == 0:
        return np.array([], dtype=bool), []
    
    n_points = len(points)
    
    print(f"  Building KD-tree for clustering...")
    tree = KDTree(points)
    
    if eps is None:
        print(f"  Calculating optimal distance threshold...")
        sample_size = min(10000, n_points)
        sample_idx = np.random.choice(n_points, sample_size, replace=False)
        distances, _ = tree.query(points[sample_idx], k=6)
        eps = np.mean(distances[:, 5]) * 1.5
    
    print(f"  Distance threshold (eps): {eps:.4f}")
    print(f"  Finding connected points...")
    
    batch_size = 5000
    rows = []
    cols = []
    
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        neighbors_list = tree.query_ball_point(points[i:end], eps)
        
        for j, neighbors in enumerate(neighbors_list):
            point_idx = i + j
            for neighbor in neighbors:
                if neighbor != point_idx:
                    rows.append(point_idx)
                    cols.append(neighbor)
        
        print_progress(end, n_points, "  Building graph")
    
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    print(f"  Finding connected components...")
    n_components, labels = connected_components(adjacency, directed=False)
    
    # Get cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_info = sorted(zip(unique_labels, counts), key=lambda x: -x[1])
    
    return labels, cluster_info, eps


def apply_cluster_filter(labels: np.ndarray, cluster_info: list, min_cluster_size: int) -> np.ndarray:
    """Apply cluster size filter and return mask."""
    large_clusters = [label for label, count in cluster_info if count >= min_cluster_size]
    return np.isin(labels, large_clusters)


def show_preview(original_count: int, stat_removed: int, cluster_info: list, min_cluster_size: int):
    """Show preview of what will be removed."""
    after_stat = original_count - stat_removed
    
    large_clusters = [(label, count) for label, count in cluster_info if count >= min_cluster_size]
    small_clusters = [(label, count) for label, count in cluster_info if count < min_cluster_size]
    
    points_kept = sum(count for _, count in large_clusters)
    points_removed_clusters = sum(count for _, count in small_clusters)
    total_removed = stat_removed + points_removed_clusters
    
    print(f"\n{'='*60}")
    print(f"PREVIEW - No changes made yet")
    print(f"{'='*60}")
    print(f"\nOriginal points:        {original_count:,}")
    print(f"After outlier removal:  {after_stat:,} (-{stat_removed:,})")
    print(f"\nCluster analysis:")
    print(f"  Total clusters found: {len(cluster_info):,}")
    print(f"  Clusters >= {min_cluster_size} pts: {len(large_clusters):,}")
    print(f"  Clusters < {min_cluster_size} pts:  {len(small_clusters):,}")
    
    # Show top clusters
    print(f"\nTop 10 largest clusters:")
    for i, (label, count) in enumerate(cluster_info[:10]):
        status = "✓ KEEP" if count >= min_cluster_size else "✗ REMOVE"
        print(f"  {i+1}. {count:,} points {status}")
    
    if len(cluster_info) > 10:
        print(f"  ... and {len(cluster_info) - 10:,} more clusters")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  Points to keep:   {points_kept:,}")
    print(f"  Points to remove: {total_removed:,} ({total_removed/original_count*100:.1f}%)")
    print(f"{'='*60}")
    
    return points_kept, total_removed


def interactive_loop(original_count: int, stat_mask: np.ndarray, filtered_points: np.ndarray,
                     labels: np.ndarray, cluster_info: list, initial_min_cluster: int, eps: float):
    """Interactive loop to adjust parameters."""
    min_cluster = initial_min_cluster
    stat_removed = original_count - len(filtered_points)
    
    while True:
        points_kept, total_removed = show_preview(original_count, stat_removed, cluster_info, min_cluster)
        
        print(f"\nOptions:")
        print(f"  [y] Yes, proceed with these settings")
        print(f"  [n] No, cancel and exit")
        print(f"  [m] Modify min-cluster size (current: {min_cluster})")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'y':
            return min_cluster, True
        elif choice == 'n':
            return min_cluster, False
        elif choice == 'm':
            try:
                new_value = input(f"Enter new min-cluster size (current: {min_cluster}): ").strip()
                new_min = int(new_value)
                if new_min < 1:
                    print("Min cluster size must be at least 1")
                else:
                    min_cluster = new_min
            except ValueError:
                print("Invalid number, please try again")
        else:
            print("Invalid choice, please enter y, n, or m")


def clean_point_cloud(input_path: str, level: str = "medium", output_path: str = None, 
                      min_cluster_size: int = 100, interactive: bool = True) -> dict:
    """Clean a PLY point cloud by removing statistical outliers and small clusters."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Will set output_path later after we know final min_cluster_size
    user_specified_output = output_path is not None
    
    params = LEVELS[level]
    
    print(f"Loading point cloud: {input_path}")
    ply_data = PlyData.read(str(input_path))
    vertex = ply_data['vertex']
    original_count = len(vertex.data)
    
    if original_count == 0:
        raise ValueError("Point cloud is empty")
    
    print(f"Original points: {original_count:,}")
    
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Step 1: Statistical outlier removal
    print(f"\n[Step 1/2] Statistical outlier removal ({level} level)")
    print(f"  Parameters: neighbors={params['nb_neighbors']}, std_ratio={params['std_ratio']}")
    stat_mask = statistical_outlier_removal(
        points,
        nb_neighbors=params["nb_neighbors"],
        std_ratio=params["std_ratio"]
    )
    stat_removed = original_count - np.sum(stat_mask)
    print(f"  Removed {stat_removed:,} statistical outliers")
    
    # Step 2: Cluster analysis
    print(f"\n[Step 2/2] Cluster analysis")
    filtered_indices = np.where(stat_mask)[0]
    filtered_points = points[stat_mask]
    
    labels, cluster_info, eps = cluster_filter_fast(filtered_points, min_cluster_size)
    
    # Interactive mode
    if interactive:
        final_min_cluster, proceed = interactive_loop(
            original_count, stat_mask, filtered_points, 
            labels, cluster_info, min_cluster_size, eps
        )
        
        if not proceed:
            print("\nCancelled. No files were modified.")
            return None
        
        min_cluster_size = final_min_cluster
    
    # Set output path with final parameters if not user-specified
    if not user_specified_output:
        output_path = input_file.parent / f"{input_file.stem}-{level}-{min_cluster_size}.ply"
    
    # Apply final filter
    cluster_mask = apply_cluster_filter(labels, cluster_info, min_cluster_size)
    
    # Combine masks
    final_indices = filtered_indices[cluster_mask]
    final_mask = np.zeros(original_count, dtype=bool)
    final_mask[final_indices] = True
    
    cleaned_vertex_data = vertex.data[final_mask]
    
    cleaned_count = len(cleaned_vertex_data)
    removed_count = original_count - cleaned_count
    removal_percent = (removed_count / original_count) * 100
    
    # Save
    cleaned_vertex = PlyElement.describe(cleaned_vertex_data, 'vertex')
    cleaned_ply = PlyData([cleaned_vertex], text=ply_data.text)
    cleaned_ply.write(str(output_path))
    
    print(f"\n✓ Saved to: {output_path}")
    
    return {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "removed_count": removed_count,
        "removal_percent": removal_percent,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean PLY point clouds by removing outliers and small clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_pointcloud.py point_cloud.ply
    python clean_pointcloud.py point_cloud.ply --level high
    python clean_pointcloud.py point_cloud.ply --level high --min-cluster 200
    python clean_pointcloud.py point_cloud.ply --no-confirm  # Skip interactive mode
        """
    )
    parser.add_argument("input", help="Input PLY file path")
    parser.add_argument(
        "--level", "-l",
        choices=["low", "medium", "high"],
        default="medium",
        help="Aggression level for outlier removal (default: medium)"
    )
    parser.add_argument(
        "--min-cluster", "-m",
        type=int,
        default=100,
        help="Minimum cluster size to keep (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output PLY file path (default: <input>_cleaned.ply)"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip interactive confirmation and proceed directly"
    )
    
    args = parser.parse_args()
    
    try:
        stats = clean_point_cloud(
            args.input, 
            args.level, 
            args.output, 
            args.min_cluster,
            interactive=not args.no_confirm
        )
        if stats:
            print("\nCleaning complete!")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
