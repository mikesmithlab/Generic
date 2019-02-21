from . import (basics, geometric, crops_and_masks, thresholding, morphological,
               drawing, contours, smoothing, features)

"""
Always import functions from the package so they can be moved around between 
modules without consequences.

If you create any new functions add them to a new or existing module and 
list them below.
"""

# Functions in basics
display = basics.display
read = basics.read_img
load = basics.read_img
write = basics.write_img
save = basics.write_img
bgr_2_grayscale = basics.bgr_2_grayscale
get_width_and_height = basics.get_width_and_height
get_width = basics.get_width
get_height = basics.get_height

# Functions in geometric
resize = geometric.resize
rotate = geometric.rotate

# Functions in crop_and_mask
crop_img = crops_and_masks.crop_img
crop_and_mask_img = crops_and_masks.crop_and_mask_image
mask_img = crops_and_masks.mask_img
CropShape = crops_and_masks.InteractiveCrop
set_edge_white = crops_and_masks.set_edge_white
mask_right = crops_and_masks.mask_right
mask_top = crops_and_masks.mask_top
imfill = crops_and_masks.imfill

# Functions in thresholding
threshold = thresholding.threshold
adaptive_threshold = thresholding.adaptive_threshold
distance_transform = thresholding.distance_transform

# Functions in morphological
dilate = morphological.dilate
erode = morphological.erode
closing = morphological.closing
opening = morphological.opening

# Functions in drawing
draw_voronoi_cells = drawing.draw_voronoi_cells
draw_polygons = drawing.draw_polygons
draw_polygon = drawing.draw_polygon
draw_delaunay_tess = drawing.draw_delaunay_tess
draw_circle = drawing.draw_circle
draw_circles = drawing.draw_circles
draw_contours = drawing.draw_contours

# Functions in contours
find_contours = contours.find_contours

# Functions in smoothing
gaussian_blur = smoothing.gaussian_blur
blur = smoothing.gaussian_blur

# Functions in features
find_circles = features.find_circles
extract_biggest_object = features.extract_biggest_object

