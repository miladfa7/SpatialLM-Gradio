

import rerun as rr
import rerun.blueprint as rrb
import numpy as np 
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose
from spatiallm import Layout


@rr.thread_local_stream("spatialLM_visualization")
def visualize_layout_with_rerun(point_cloud_path, layout_content, radius=0.01, max_points=1000000):
    stream = rr.binary_stream()
    try:        
        # Create a 3D visualization blueprint
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        layout = Layout(layout_content)
        floor_plan = layout.to_boxes()
        
        print("Loading point cloud")
        pcd = load_o3d_pcd(point_cloud_path)
        points, colors = get_points_and_colors(pcd)
        
        print(f"Point cloud loaded: {len(points)} points")
        
        if points.shape[0] > max_points:
            point_indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[point_indices]
            colors = colors[point_indices]
            print(f"Subsampled to {len(points)} points")
        
        print("Logging points to Rerun")
        rr.log(
            "world/points",
            rr.Points3D(
                positions=points,
                colors=colors,
                radii=radius,
            ),
            static=True,
        )
        yield stream.read()
        
        for ti, box in enumerate(floor_plan):
            uid = box["id"]
            group = box["class"]
            label = box["label"]
            
            rr.set_time_sequence("box", ti)
            rr.log(
                f"world/pred/{group}/{uid}",
                rr.Boxes3D(
                    centers=box["center"],
                    half_sizes=0.5 * box["scale"],
                    labels=label,
                ),
                rr.InstancePoses3D(mat3x3=box["rotation"]),
                static=False,
            )
            yield stream.read()
                
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    yield stream.read()
