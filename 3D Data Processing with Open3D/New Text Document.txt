# %% [markdown]
# # 3D Data Processing with Open3D

# %% [markdown]
# This notebook provides a quick walkthrough on how to explore, process and visualize a 3D model using Python's [Open3D](http://www.open3d.org/docs/release/index.html) library. The specific 3D data processing tasks discussed in this notebook are as follows:
# 1. [Loading and visualizing a 3D model as *mesh*](#Loading-and-visualizing-a-3D-model-as-mesh)
# 2. [Converting *mesh* to *point cloud* by sampling points](#Converting-mesh-to-point-cloud-by-sampling-points)
# 3. [Removing hidden points from *point cloud*](#Removing-hidden-points-from-point-cloud)
# 4. [Converting *point cloud* to dataframe](#Converting-point-cloud-to-dataframe)
# 5. [Saving the *point cloud* and dataframe](#Saving-the-point-cloud-and-dataframe)

# %%
# Importing open3d library.

import open3d as o3d

# %%
# Checking the installed version.

o3d.__version__
# Open3D version used in this exercise: 0.16.0

# %%
# Importing all other necessary libraries.

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)

# %% [markdown]
# ## Loading and visualizing a 3D model as *mesh*

# %% [markdown]
# The 3D model used in this notebook has been modified slightly from the original file to fit the purposes of this exercise. 
# 
# Credit to the original creator - "Tesla Model S Plaid" (https://skfb.ly/oEqT9) by ValentunW is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).

# %%
# Defining the path to the 3D model file.

mesh_path = "data/3d_model.obj"

# %%
# Reading the 3D model file as a 3D mesh using open3d.

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh

# %%
# Visualizing the mesh.

draw_geoms_list = [mesh]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# The car mesh does not appear 3D and is painted in a uniform grey colour. This is because the mesh does not have any information about *normals* for the vertices and the surfaces in the 3D model.

# %% [markdown]
# **What are *normals*?** - The normal vector to a surface at a given point is a vector which is perpendicular to the surface at that point. The normal vector is often simply called the "*normal*". Check out these links for a more detailed explanation on this topic:
# - [Normal Vector](https://mathworld.wolfram.com/NormalVector.html)
# - [Estimating Surface Normals in a PointCloud](https://pcl.readthedocs.io/en/latest/normal_estimation.html#)
# 
# ![Normal vector](assets/Normal_Vector.png "Normal to a surface at a given point")

# %%
# Computing the normals for the mesh.

mesh.compute_vertex_normals()
mesh

# %%
# Visualizing the mesh with the estimated surface normals.

draw_geoms_list = [mesh]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# After computing the normals, the car renders properly and looks like a 3D model.

# %%
# Creating a mesh of the XYZ axes Cartesian coordinates frame. This mesh will show the directions in which the X, Y & Z-axes point, and 
# can be overlaid on the 3D mesh to visualize its orientation in the Euclidean space.
# X-axis : Red arrow
# Y-axis : Green arrow
# Z-axis : Blue arrow

mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
mesh_coord_frame

# %%
# Visualizing the mesh with the coordinate frame to understand the orientation.

draw_geoms_list = [mesh_coord_frame, mesh]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# From the above visualization, we see that the car is oriented as follows:
# - **Origin** of XYZ axes : **At the volumetric center** of the car model.
# - **X-axis** (Red arrow) : Along the **length dimension** of the car with positive X-axis pointing towards the hood of the car.
# - **Y-axis** (Green arrow) : Along the **height dimension** of the car with the positive Y-axis pointing towards the roof of the car.
# - **Z-axis** (Blue arrow) : Along the **width dimension** of the car with the positive Z-axis pointing towards the right side of the car.

# %% [markdown]
# Let us now take a look at what is inside this car model. We will crop the mesh in the Z-axis and remove the right half of the car (positive Z-axis).

# %%
# Cropping the car mesh using its bouding box to remove its right half (positive Z-axis).

bbox = mesh.get_axis_aligned_bounding_box()
bbox_points = np.asarray(bbox.get_box_points())
bbox_points[:, 2] = np.clip(bbox_points[:, 2], a_min=None, a_max=0)
bbox_cropped = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_points))
mesh_cropped = mesh.crop(bbox_cropped)
mesh_cropped

# %%
# Visualizing the cropped mesh.

draw_geoms_list = [mesh_coord_frame, mesh_cropped]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# From the above visualization, we see that this car model has a detailed interior. Now that we have seen what is inside this 3D mesh, we can convert it to a point cloud before removing the "*hidden*" points which belong to the interior of the car.

# %% [markdown]
# ## Converting *mesh* to *point cloud* by sampling points

# %% [markdown]
# Converting the mesh to a point cloud can be easily done in Open3D by defining the number of points we wish to sample from the mesh.

# %%
# Uniformly sampling 100,000 points from the mesh to convert it to a point cloud.

n_pts = 100_000
pcd = mesh.sample_points_uniformly(n_pts)
pcd

# %%
# Visualizing the mesh and the point cloud together.

draw_geoms_list = [mesh_coord_frame, mesh, pcd]
o3d.visualization.draw_geometries(draw_geoms_list)

# %%
# Visualizing the point cloud.

draw_geoms_list = [mesh_coord_frame, pcd]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# Take note that the colours in the point cloud visualization above only indicate the position of the points along the Z-axis.

# %%
# Cropping the car point cloud using bounding box to remove its right half (positive Z-axis).

pcd_cropped = pcd.crop(bbox_cropped)
pcd_cropped

# %%
# Visualizing the cropped point cloud.

draw_geoms_list = [mesh_coord_frame, pcd_cropped]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# We see from the above visualization of the cropped point cloud that it also contains points which belong to the interior of the car model. This is expected as this point cloud was created by uniformly sampling points from the entire mesh. In the next section, we will remove these "*hidden*" points which belong to the interior of the car and are not on the outer surface of the point cloud.

# %% [markdown]
# ## Removing hidden points from *point cloud*

# %% [markdown]
# Imagine yourself pointing a light on the right side of the car model. All the points that fall on the right outer surface of the 3D model would be illuminated, while all the other points in the point cloud would not.
# 
# ![Hidden point removal view 1 illustration](assets/HPR_Demo_View_1_Illustration.jpg "How Open3D's Hidden Point Removal works on the point cloud from a given viewpoint. Illuminated points are considered 'visible', all other points are considered 'hidden'.")
# 
# We can now label these illuminated points as "*visible*" and all the non-illuminated points as "*hidden*". These "*hidden*" points would also include all the points that belong to the interior of the car. This operation is known as **Hidden Point Removal** in Open3D. Check out [this link](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal) for an example on hidden point removal in Open3D documentation.

# %%
# Defining the camera and radius parameters for the hidden point removal operation.

diameter = np.linalg.norm(np.asarray(pcd.get_min_bound()) - np.asarray(pcd.get_max_bound()))
camera = [0, 0, diameter]
radius = diameter * 100

print(camera)
print(radius)

# %%
# Performing the hidden point removal operation on the point cloud using the camera and radius parameters defined above.
# The output is a list of indexes of points that are not hidden.

_, pt_map = pcd.hidden_point_removal(camera, radius)
pt_map

# %% [markdown]
# Using the above output list of indexes of points that are visible, we can colour the visible and hidden points in different colours before visualizing the point cloud.

# %%
# Painting all the visible points in the point cloud in blue, and all the hidden points in red.

pcd_visible = pcd.select_by_index(pt_map)
pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
print("No. of visible points : ", pcd_visible)

pcd_hidden = pcd.select_by_index(pt_map, invert=True)
pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
print("No. of hidden points : ", pcd_hidden)

# %%
# Visualizing the visible (blue) and hidden (red) points in the point cloud.

draw_geoms_list = [mesh_coord_frame, pcd_visible, pcd_hidden]
# draw_geoms_list = [mesh_coord_frame, pcd_visible]
# draw_geoms_list = [mesh_coord_frame, pcd_hidden]

o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# From the visualization above, we see how the hidden point removal operation works from a given camera viewpoint. The operation eliminates all the points in the background that are occluded by the points in the foreground from a given camera viewpoint.
# 
# To understand this better, let us see how the same operation would work if we were to rotate the point cloud slightly. **Effectively, we're trying to change the viewpoint here. But instead of changing it by re-defining the camera parameters, we're rotating the point cloud itself.**

# %%
# Defining a function to convert degrees to radians.

def deg2rad(deg):
    return deg * np.pi/180

# %%
# Rotating the point cloud about the X-axis by 90 degrees.

x_theta = deg2rad(90)
y_theta = deg2rad(0)
z_theta = deg2rad(0)

tmp_pcd_r = copy.deepcopy(pcd)
R = tmp_pcd_r.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
tmp_pcd_r.rotate(R, center=(0, 0, 0))
tmp_pcd_r

# %%
# Visualizing the rotated point cloud.

draw_geoms_list = [mesh_coord_frame, tmp_pcd_r]
o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# By repeating the same process again with the rotated car model, we would see that this time all the points that fall on the upper outer surface of the 3D model (roof of the car) would get illuminated, while all the other points in the point cloud would not.
# 
# ![Hidden point removal view 2 illustration](assets/HPR_Demo_View_2_Illustration.jpg "Hidden Point Removal on the rotated point cloud from the same viewpoint as earlier. Illuminated points are considered 'visible', all other points are considered 'hidden'.")

# %%
# Performing the hidden point removal operation on the rotated point cloud using the same camera and radius parameters
# defined above. The output is a list of indexes of points that are not hidden.

_, pt_map = tmp_pcd_r.hidden_point_removal(camera, radius)
pt_map

# %%
# Painting all the visible points in the rotated point cloud in blue, and all the hidden points in red.

pcd_visible = tmp_pcd_r.select_by_index(pt_map)
pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
print("No. of visible points : ", pcd_visible)

pcd_hidden = tmp_pcd_r.select_by_index(pt_map, invert=True)
pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
print("No. of hidden points : ", pcd_hidden)

# %%
# Visualizing the visible (blue) and hidden (red) points in the rotated point cloud.

draw_geoms_list = [mesh_coord_frame, pcd_visible, pcd_hidden]
# draw_geoms_list = [mesh_coord_frame, pcd_visible]
# draw_geoms_list = [mesh_coord_frame, pcd_hidden]

o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# The above visualization of the rotated point cloud clearly illustrates how the hidden point removal operation works. So, now in order to remove *all* the "*hidden*" points from this car point cloud, we can **perform this hidden point removal operation *sequentially* by rotating the point cloud slightly in all the three axes from -90 to +90 degrees.** After each hidden point removal operation, we can aggregate the output list of indexes of points. **After all the hidden point removal opertions, the aggregated list of indexes of points will contain all the points that are not hidden (ie., points that are on the outer surface of the point cloud).**

# %%
# Defining a function to rotate a point cloud in X, Y and Z-axis.

def get_rotated_pcd(pcd, x_theta, y_theta, z_theta):

    pcd_rotated = copy.deepcopy(pcd)
    R = pcd_rotated.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    pcd_rotated.rotate(R, center=(0, 0, 0))
    
    return pcd_rotated

# %%
# Defining a function to get the camera and radius parameters for the point cloud for the hidden point removal operation.

def get_hpr_camera_radius(pcd):
    
    diameter = np.linalg.norm(np.asarray(pcd.get_min_bound()) - np.asarray(pcd.get_max_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100
    
    return camera, radius

# %%
# Defining a function to perform the hidden point removal operation on the point cloud using the camera and radius parameters
# defined earlier. The output is a list of indexes of points that are not hidden.

def get_hpr_pt_map(pcd, camera, radius):

    _, pt_map = pcd.hidden_point_removal(camera, radius)    
    return pt_map

# %%
# Performing the hidden point removal operation sequentially by rotating the point cloud slightly in each of the three axes
# from -90 to +90 degrees, and aggregating the list of indexes of points that are not hidden after each operation.

# Defining a list to store the aggregated output lists from each hidden point removal operation.
pt_map_aggregated = []

# Defining the steps and range of angle values by which to rotate the point cloud.
theta_range = np.linspace(-90, 90, 7)

# Counting the number of sequential operations.
view_counter = 1
total_views = theta_range.shape[0] ** 3

# Obtaining the camera and radius parameters for the hidden point removal operation.
camera, radius = get_hpr_camera_radius(pcd)

# Looping through the angle values defined above for each axis.
for x_theta_deg in theta_range:
    for y_theta_deg in theta_range:
        for z_theta_deg in theta_range:

            print(f"Removing hidden points - processing view {view_counter} of {total_views}.")

            # Rotating the point cloud by the given angle values.
            x_theta = deg2rad(x_theta_deg)
            y_theta = deg2rad(y_theta_deg)
            z_theta = deg2rad(z_theta_deg)
            pcd_rotated = get_rotated_pcd(pcd, x_theta, y_theta, z_theta)
            
            # Performing the hidden point removal operation on the rotated point cloud using the camera and radius parameters
            # defined above.
            pt_map = get_hpr_pt_map(pcd_rotated, camera, radius)
            
            # Aggregating the output list of indexes of points that are not hidden.
            pt_map_aggregated += pt_map

            view_counter += 1

# Removing all the duplicated points from the aggregated list by converting it to a set.
pt_map_aggregated = list(set(pt_map_aggregated))

# %%
pt_map_aggregated

# %%
# Painting all the visible points in the point cloud in blue, and all the hidden points in red.

pcd_visible = pcd.select_by_index(pt_map_aggregated)
pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
print("No. of visible points : ", pcd_visible)

pcd_hidden = pcd.select_by_index(pt_map_aggregated, invert=True)
pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
print("No. of hidden points : ", pcd_hidden)

# %%
# Visualizing the visible (blue) and hidden (red) points in the point cloud.

draw_geoms_list = [mesh_coord_frame, pcd_visible, pcd_hidden]
# draw_geoms_list = [mesh_coord_frame, pcd_visible]
# draw_geoms_list = [mesh_coord_frame, pcd_hidden]

o3d.visualization.draw_geometries(draw_geoms_list)

# %%
# Cropping the point cloud of visible points using bounding box to remove its right half (positive Z-axis).

pcd_visible_cropped = pcd_visible.crop(bbox_cropped)
pcd_visible_cropped

# %%
# Cropping the point cloud of hidden points using bounding box to remove its right half (positive Z-axis).

pcd_hidden_cropped = pcd_hidden.crop(bbox_cropped)
pcd_hidden_cropped

# %%
# Visualizing the cropped point clouds.

draw_geoms_list = [mesh_coord_frame, pcd_visible_cropped, pcd_hidden_cropped]
# draw_geoms_list = [mesh_coord_frame, pcd_visible_cropped]
# draw_geoms_list = [mesh_coord_frame, pcd_hidden_cropped]

o3d.visualization.draw_geometries(draw_geoms_list)

# %% [markdown]
# From the above visualization of the cropped point cloud after the hidden point removal operation, we see that all the "*hidden*" points which belong to the interior of the car model (red) are now separated from the "*visible*" points which are on the outer surface of the point cloud (blue).

# %% [markdown]
# ## Converting *point cloud* to dataframe

# %% [markdown]
# As one might expect, the position of each point in the point cloud can be defined by three numerical values - the X, Y & Z coordinates. However, recall that in the section above, we also estimated the surface normals for each point in the 3D mesh. As we sampled points from this mesh to create the point cloud, each point in the point cloud also contains three additional attributes related to these surface normals - the normal unit vector coordinates in the X, Y & Z directions.
# 
# So, in order to convert the point cloud to a dataframe, **each point in the point cloud can be represented in a single row by the following seven attribute columns**:
# 1. **X coordinate** (*float*)
# 2. **Y coordinate** (*float*)
# 3. **Z coordinate** (*float*)
# 4. **Normal vector coordinate in X direction** (*float*)
# 5. **Normal vector coordinate in Y direction** (*float*)
# 6. **Normal vector coordinate in Z direction** (*float*)
# 7. **Point visible** (*boolean True or False*)

# %%
# Accessing the X, Y & Z positional coordinates of all the points in the point cloud.

np.asarray(pcd.points)

# %%
np.asarray(pcd.points).shape

# %%
# Accessing the normal unit vector coordinates in the X, Y & Z directions of all the points in the point cloud.

np.asarray(pcd.normals)

# %%
np.asarray(pcd.normals).shape

# %%
# Creating a dataframe for the point cloud with the X, Y & Z positional coordinates and the normal unit vector coordinates
# in the X, Y & Z directions of all points.

pcd_df = pd.DataFrame(np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                      columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"]
                     )
pcd_df

# %%
# Adding a column to indicate whether the point is visible or not using the aggregated list of indexes of points from the 
# hidden point removal operation above.

pcd_df["point_visible"] = False
pcd_df.loc[pt_map_aggregated, "point_visible"] = True
pcd_df

# %%
# Checking the numbers of hidden and visible points in the point cloud.

pcd_df["point_visible"].value_counts()

# %%
# Checking the percentages of hidden and visible points in the point cloud.

pcd_df["point_visible"].value_counts(normalize=True)

# %% [markdown]
# ## Saving the *point cloud* and dataframe

# %%
# Saving the entire point cloud as a .pcd file.

pcd_save_path = "data/3d_model.pcd"
o3d.io.write_point_cloud(pcd_save_path, pcd)

# %%
# Saving the point cloud with the hidden points removed as a .pcd file.

pcd_visible_save_path = "data/3d_model_hpr.pcd"
o3d.io.write_point_cloud(pcd_visible_save_path, pcd_visible)

# %%
# Saving the point cloud dataframe as a .csv file.

pcd_df_save_path = "data/3d_model.csv"
pcd_df.to_csv(pcd_df_save_path, index=False)

# %%


# %% [markdown]
# # Extra Content

# %% [markdown]
# ## Some useful function definitions

# %%
def read_mesh(mesh_path, compute_normals=True):
    
    """Read a 3D mesh using Open3D.
    
    Args:
        mesh_path: Source path to the 3D mesh file (.obj, .gltf, .glb, etc.).
        compute_normals: (bool, default=True) Whether to estimate surface normals for the 3D mesh or not.
        
    Returns:
        mesh: The read 3D mesh as a open3d.geometry.TriangleMesh object.
    """
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if compute_normals:
        mesh = mesh.compute_vertex_normals()
    return mesh

# %%
def get_mesh_mmc_coords(mesh, coords_type="centre"):
    
    """Get the centre, minimum or maximum XYZ coordinates of a 3D mesh.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        coords_type: (string, default="centre") Either "centre", "min" or "max" to get the corresponding XYZ coordinates.
        
    Returns:
        An array of the corresponding XYZ coordinates from the 3D mesh.
    """
        
    if coords_type == "centre":
        return mesh.get_center()
    elif coords_type == "max":
        return mesh.get_max_bound()
    elif coords_type == "min":
        return mesh.get_min_bound()

# %%
def get_mesh_xyz_range(mesh):
    
    """Get the range of XYZ coordinates of a 3D mesh.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        
    Returns:
        An array of the corresponding range of XYZ coordinates from the 3D mesh.
    """
    
    return get_mesh_mmc_coords(mesh, "max") - get_mesh_mmc_coords(mesh, "min")

# %%
def get_translated_mesh(mesh, x_tr, y_tr, z_tr):
    
    """Translate the 3D mesh along the X, Y & Z-axes.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        x_tr: (float) A value to move the mesh in X-axis.
        y_tr: (float) A value to move the mesh in Y-axis.
        z_tr: (float) A value to move the mesh in Z-axis.
        
    Returns:
        mesh_translated: The translated 3D mesh as a open3d.geometry.TriangleMesh object.
    """
        
    mesh_translated = copy.deepcopy(mesh)
    mesh_translated.translate([x_tr, y_tr, z_tr])
    
    return mesh_translated

# %%
def get_rotated_mesh(mesh, x_theta, y_theta, z_theta):
    
    """Rotate the 3D mesh about the X, Y & Z-axes.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        x_theta: (float) An angle value in radians to rotate the mesh about the X-axis.
        y_theta: (float) An angle value in radians to rotate the mesh about the Y-axis.
        z_theta: (float) An angle value in radians to rotate the mesh about the Z-axis.
        
    Returns:
        mesh_rotated: The rotated 3D mesh as a open3d.geometry.TriangleMesh object.
    """
    
    mesh_rotated = copy.deepcopy(mesh)
    R = mesh_rotated.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    mesh_rotated.rotate(R, center=(0, 0, 0))
    
    return mesh_rotated


def deg2rad(deg):
    return deg * np.pi/180

# %%
def get_scaled_mesh(mesh, scale_factor):
    
    """Scale the 3D mesh by a scaling factor.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        scale_factor: (float) A factor value to scale the size of the mesh.
        
    Returns:
        mesh_scaled: The scaled 3D mesh as a open3d.geometry.TriangleMesh object.
    """
    
    mesh_scaled = copy.deepcopy(mesh)
    mesh_scaled.scale(scale_factor, center=(0, 0, 0))
    
    return mesh_scaled

# %%
def get_painted_mesh(mesh, colour_rgb):
    
    """Paint the 3D mesh in a uniform colour.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        colour_rgb: (array) An array containing the RGB values of a colour normalized between 0 and 1.
        
    Returns:
        mesh_painted: The painted 3D mesh as a open3d.geometry.TriangleMesh object.
    """
    
    mesh_painted = copy.deepcopy(mesh)
    mesh_painted.paint_uniform_color(colour_rgb)
    
    return mesh_painted

# %%
def save_mesh(mesh, mesh_path):
    
    """Save a 3D mesh using Open3D.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        mesh_path: Destination path for the 3D mesh file (.obj, .gltf, .glb, etc.). If path does not exist, it is created.
    """
    
    if not os.path.exists(os.path.split(mesh_path)[0]):
        os.makedirs(os.path.split(mesh_path)[0])
    
    o3d.io.write_triangle_mesh(mesh_path, mesh)

# %%
def sample_points_from_mesh(mesh, n_pts):
    
    """Create a 3D point cloud by uniformly sampling points from a 3D mesh using Open3D.
    
    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        n_pts: (integer) Number of points to be sampled from the 3D mesh.
        
    Returns:
        pcd: The created 3D point cloud as a open3d.geometry.PointCloud object.
    """
    
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd

# %%
def read_pcd(pcd_path):
    
    """Read a 3D point cloud using Open3D.
    
    Args:
        pcd_path: Source path to the 3D point cloud file (.pcd, .ply, .pts, etc.).
        
    Returns:
        pcd: The read 3D point cloud as a open3d.geometry.PointCloud object.
    """
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

# %%
def get_pcd_mmc_coords(pcd, coords_type="centre"):
    
    """Get the centre, minimum or maximum XYZ coordinates of a 3D point cloud.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        coords_type: (string, default="centre") Either "centre", "min" or "max" to get the corresponding XYZ coordinates.
        
    Returns:
        (array) An array of the corresponding XYZ coordinates from the 3D point cloud.
    """
    
    if coords_type == "centre":
        return pcd.get_center()
    elif coords_type == "max":
        return pcd.get_max_bound()
    elif coords_type == "min":
        return pcd.get_min_bound()

# %%
def get_rotated_pcd(pcd, x_theta, y_theta, z_theta):
    
    """Rotate the 3D point cloud about the X, Y & Z-axes.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        x_theta: (float) An angle value in radians to rotate the mesh about the X-axis.
        y_theta: (float) An angle value in radians to rotate the mesh about the Y-axis.
        z_theta: (float) An angle value in radians to rotate the mesh about the Z-axis.
        
    Returns:
        pcd_rotated: The rotated 3D point cloud as a open3d.geometry.PointCloud object.
    """
    
    pcd_rotated = copy.deepcopy(pcd)
    R = pcd_rotated.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    pcd_rotated.rotate(R, center=(0, 0, 0))
    
    return pcd_rotated

# %%
def get_painted_pcd(pcd, colour_rgb):
    
    """Paint the 3D point cloud in a uniform colour.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        colour_rgb: (array) An array containing the RGB values of a colour normalized between 0 and 1.
        
    Returns:
        pcd_painted: The painted 3D point cloud as a open3d.geometry.PointCloud object.
    """
    
    pcd_painted = copy.deepcopy(pcd)
    pcd_painted.paint_uniform_color(colour_rgb)
    
    return pcd_painted

# %%
def convert_pcd2df(pcd):
    
    """Convert the 3D point cloud to a dataframe. Each row in the dataframe will represent a point in the 3D point cloud.
    The dataframe will contain 6 columns - the X, Y & Z positional coordinates and the normal unit vector coordinates
    in the X, Y & Z directions of all points.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        
    Returns:
        pcd_df: The created dataframe as a pandas.DataFrame object.
    """
    
    pcd_df = pd.DataFrame(np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"]
                         )
    return pcd_df

# %%
def convert_df2pcd(pcd_df):
    
    """Convert the dataframe to a 3D point cloud. Each row in the dataframe must represent a point for the 3D point cloud.
    The dataframe must contain 6 columns - the X, Y & Z positional coordinates and the normal unit vector coordinates
    in the X, Y & Z directions of all points.
    
    Args:
        pcd_df: A pandas.DataFrame object.
        
    Returns:
        pcd: The created 3D point cloud as an open3d.geometry.PointCloud object.
    """
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_df[["x", "y", "z"]]))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd_df[["norm-x", "norm-y", "norm-z"]]))
    
    return pcd

# %%
def get_hpr_camera_radius(pcd):
    
    """Obtain the camera and radius parameters to define the camera viewpoint for the hidden point removal operation
    based on the dimensions of the 3D point cloud.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        
    Returns:
        camera: (list of floats) A list of the corresponding XYZ coordinates for the camera position.
        radius: (float) The radius parameter for the camera viewpoint.
    """
    
    diameter = np.linalg.norm(np.asarray(get_pcd_mmc_coords(pcd, "min")) - np.asarray(get_pcd_mmc_coords(pcd, "max")))
    camera = [0, 0, diameter]
    radius = diameter * 100
    
    return camera, radius


def get_hpr_pt_map(pcd, camera, radius):
    
    """Perform the hidden point removal operation on the 3D point cloud from the given camera viewpoint.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        camera: (list of floats) A list of the XYZ coordinates for the camera position.
        radius: (float) The radius parameter for the camera viewpoint.
        
    Returns:
        pt_map: (list of integers) A list of indexes of points from the 3D point cloud that are visible from the given
                camera viewpoint.
    """
    
    _, pt_map = pcd.hidden_point_removal(camera, radius)    
    return pt_map

# %%
def save_pcd(pcd, pcd_path):
    
    """Saves a 3D point cloud using Open3D.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        pcd_path: Destination path for the 3D point cloud file (.pcd, .ply, .pts, etc.). If path does not exist, it is created.
    """
    
    if not os.path.exists(os.path.split(pcd_path)[0]):
        os.makedirs(os.path.split(pcd_path)[0])
    
    o3d.io.write_point_cloud(pcd_path, pcd)

# %%
def save_df_as_csv(df, df_path):
    
    """Saves a dataframe as csv.
    
    Args:
        df: A pandas.DataFrame object.
        df_path: Destination path for the dataframe .csv file. If path does not exist, it is created.
    """
    
    if not os.path.exists(os.path.split(df_path)[0]):
        os.makedirs(os.path.split(df_path)[0])
    
    df.to_csv(df_path, index=False)

# %% [markdown]
# ## Modifying the original 3D model

# %% [markdown]
# The original 3D model's .obj file can be downloaded from [this link](https://sketchfab.com/3d-models/tesla-model-s-plaid-9de8855fae324e6cbbb83c9b5288c961). This 3D model will be modified slightly in this section to fit the purposes of this notebook, and the modified model will be saved as a separate .obj file.
# 
# Credit to the original creator - "Tesla Model S Plaid" (https://skfb.ly/oEqT9) by ValentunW is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).

# %%
# Defining the path to the original 3D model file.

mesh_path = "data/tesla_model_s_plaid.obj"

# %%
# Reading the 3D model file as a 3D mesh and computing the normals.

mesh = read_mesh(mesh_path, compute_normals=True)
mesh

# %%
# Creating a mesh of the XYZ axes Cartesian coordinates frame. This mesh will show the directions in which the X, Y & Z-axes point, and 
# can be overlaid on the 3D mesh to visualize its orientation in the Euclidean space.
# X-axis : Red arrow
# Y-axis : Green arrow
# Z-axis : Blue arrow

mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
mesh_coord_frame

# %%
# Visualizing the mesh.

draw_geoms_list = [mesh_coord_frame, mesh]
o3d.visualization.draw_geometries(draw_geoms_list)

# %%
# Translating the mesh along the X, Y & Z-axes to move the XYZ origin to the volumetric center of the car model.

mesh_centre_coords = get_mesh_mmc_coords(mesh, "centre")
x_tr, y_tr, z_tr = -mesh_centre_coords
mesh_tr = get_translated_mesh(mesh, x_tr, y_tr, z_tr)
mesh_tr

# %%
# Visualizing the translated mesh.

draw_geoms_list = [mesh_coord_frame, mesh_tr]
o3d.visualization.draw_geometries(draw_geoms_list)

# %%
# Rotating the mesh about the Y-axis by 180 degrees to align the X & Z-axes in the correct directions.

x_theta = deg2rad(0)
y_theta = deg2rad(180)
z_theta = deg2rad(0)

mesh_tr_r = get_rotated_mesh(mesh_tr, x_theta, y_theta, z_theta)
mesh_tr_r

# %%
# Visualizing the rotated mesh.

draw_geoms_list = [mesh_coord_frame, mesh_tr_r]
o3d.visualization.draw_geometries(draw_geoms_list)

# %%
# Saving the translated and rotated mesh as a .obj file.

mesh_save_path = "data/3d_model.obj"
save_mesh(mesh_tr_r, mesh_save_path)

# %%
# Getting the bouding box XYZ coordinates for the mesh.

bbox = mesh_tr_r.get_axis_aligned_bounding_box()
bbox

# %%
# Accessing the eight XYZ points that define the bounding box.

bbox_points = np.asarray(bbox.get_box_points())
bbox_points

# %%
# Cropping the bounding box in Z-axis by limiting the Z-axis coordinate maximum values to 0.

bbox_points[:, 2] = np.clip(bbox_points[:, 2], a_min=None, a_max=0)
bbox_points

# %%
# Creating a new bounding box object with the cropped coordinates from above.

bbox_cropped = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_points))
bbox_cropped

# %%
# Cropping the mesh using the cropped bounding box created above.

mesh_tr_r_crop = mesh_tr_r.crop(bbox_cropped)

# %%
# Visualizing the cropped mesh.

draw_geoms_list = [mesh_coord_frame, mesh_tr_r_crop]
o3d.visualization.draw_geometries(draw_geoms_list, mesh_show_back_face=False)

# %% [markdown]
# ## Visualizations with animation

# %%
# Defining a function to convert degrees to radians.

def deg2rad(deg):
    return deg * np.pi/180

# %%
# Defining a function to create a custom animation for the visualizations.

def custom_animation(vis):
    
    global i
    global imgs_list
    
    if i == 0:
        ctr = vis.get_view_control()
        ctr.scale(-5)
        i+=1
        return False
    
    elif i <= 446:
        y = 4*np.sin(deg2rad((i-1)/(223/360)))
        
        ctr = vis.get_view_control()
        ctr.rotate(-5.0, y)
        
    else:
        vis.close()
    
    # vis.capture_screen_image(f"images/sample_image_{i}.png", do_render=False)
    img = vis.capture_screen_float_buffer()
    img = (255 * np.asarray(img)).astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    imgs_list.append(img)
    
    i += 1
    
    return False

# %%
# Visualizing the mesh with the custom animation function defined above.

i = 0
imgs_list = []
# draw_geoms_list = [mesh_coord_frame, mesh_tr_r_crop]
o3d.visualization.draw_geometries_with_animation_callback(draw_geoms_list, custom_animation)

# %%
# Saving all the frames of the animated visualization as .png files.

for i, img in enumerate(imgs_list):
    img.save(f"images/sample_image_{i}.png")

# %%
# Saving all the frames of the animated visualization as a .gif file.

imgs_list[0].save("images/sample_image.gif",
                  save_all=True,
                  append_images=imgs_list[1:],
                  duration=40,
                  loop=0
                 )

# %%



