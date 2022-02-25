# %% 
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from cmath import inf
from numba import jit
import cv2

# %%
# Vector support functions
# ----------------------------------------------


@jit(nopython=True)
def vec2(x, y):
    return np.array([x, y])


@jit(nopython=True)
def vec3(x, y, z):
    return np.array([x, y, z])


@jit(nopython=True)
def vec4(x, y, z, w):
    return np.array([x, y, z, w])


# %%
# Conveniance functions
# ----------------------------------------------

LINE = 0
RECT = 1
CIRCLE = 2
POLYGON = 3
    
    
def generate_render_texture(width, height, depth=4, type=np.uint8):
    return np.full((height, width, depth), fill_value=255, dtype=np.uint8)


def save_tex(texture, name="image.png"):
    Image.fromarray(texture).convert("RGBA").save(name)


def show_image(texture):
    Image.fromarray(texture).show("SDF Renderer")


@jit(nopython=True)
def clamp(x, lower, upper):
    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


@jit(nopython=True)
def smoothstep(x, e1, e2):
    x = clamp((x - e1) / (e2 - e1), 0.0, 1.0)
    return x * x * (3 - 2 * x)


@jit(nopython=True)
def dot(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


@jit(nopython=True)
def length(x1, y1):
    return math.sqrt(x1 ** 2 + y1 ** 2)


@jit(nopython=True)
def inflate(distance, amount):
    return distance - amount


@jit(nopython=True)
def deflate(distance, amount):
    return distance + amount


@jit(nopython=True)
def interpolate_colors(c1, c2, t):
    # t: float[0..1]
    # c1, c2: int[0...255, 0...255, 0...255, 0...255]
    r = c1[0] * (1 - t) + c2[0] * t
    g = c1[1] * (1 - t) + c2[1] * t
    b = c1[2] * (1 - t) + c2[2] * t
    a = c1[3] * (1 - t) + c2[3] * t
    return [r, g, b, a]


# %%
# SDF function of basic shapes
# ----------------------------------------------

@jit(nopython=True)
def line(sample_x, sample_y, startx, starty, endx, endy, radius):
    pax, pay = sample_x - startx, sample_y - starty
    bax, bay = endx - startx, endy - starty

    h = min(1.0, max(0.0, (pax * bax + pay * bay) / (bax * bax + bay * bay)))

    return length(sample_x - startx - bax * h, sample_y - starty - bay * h) - radius

# TODO: Implement line bounding box, dont really know how to implement 

@jit(nopython=True)
def box(sample_x, sample_y, bx, by, bw, bh):
    rx, ry = bw / 2, bh / 2
    qx = abs(sample_x - bx) - rx
    qy = abs(sample_y - by) - ry

    return length(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)

@jit(nopython=True)
def rounded_box(sample_x, sample_y, bx, by, bw, bh, radius):
    return inflate(box(sample_x, sample_y, bx, by, bw - 2 * radius, bh - 2 * radius), radius)

@jit(nopython=True)
def circle(sample_x, sample_y, r, cx, cy):
    return math.sqrt((sample_x - cx) ** 2 + (sample_y - cy) ** 2) - r

@jit(nopython=True)
def polygon(sample_x, sample_y, points):
    n = len(points)
    d = dot(sample_x - points[0][0], sample_y - points[0][1], sample_x - points[0][0], sample_y - points[0][1])
    s = 1.0
    for i in range(n):
        j = (i + n - 1) % n

        ex = points[j][0] - points[i][0]
        ey = points[j][1] - points[i][1]

        wx = sample_x - points[i][0]
        wy = sample_y - points[i][1]

        dd = dot(wx, wy, ex, ey) / dot(ex, ey, ex, ey)
        cl = clamp( dd, 0.0, 1.0)

        bx = wx - ex * cl
        by = wy - ey * cl
        
        d = min(d, dot(bx, by, bx, by))

        cond_x = sample_y >= points[i][1]
        cond_y = sample_y < points[j][1]
        cond_z = ex * wy > ey * wx

        if((cond_x and cond_y and cond_z) or (not cond_x and not cond_y and not cond_z)):
            s = -s

    return s * math.sqrt(d)

@jit(nopython=True)
def multiply(c, f):
    r = c[0] * f
    g = c[1] * f
    b = c[2] * f
    return [r, g, b, 255]

@jit(nopython=True)
def render(render_tex, tex_width, tex_height, objects):
    for w in range(tex_width):
        for h in range(tex_height):
            for object_descriptor in objects:
                d = inf
                if object_descriptor[0] == LINE:
                    d = line(w, h, 10, 10, 1910, 1070, 2)
                if object_descriptor[0] == RECT:
                    d = rounded_box(w, h, 1200, 540, 500, 700, 15)
                if object_descriptor[0] == CIRCLE:
                    d = circle(w, h, 200, 800, 500)
                if object_descriptor[0] == POLYGON:
                    d = inflate(polygon(w, h, [[250, 250], [650, 250], [650, 550], [400, 550], [250, 400]]), 15)
                
                if d == inf:
                    continue

                shadow_angle = np.pi
                elevation = 10

                if d < -0.5:
                    render_tex[h][w] = [object_descriptor[1], object_descriptor[2], object_descriptor[3], 255]

                if 1 >= d >= -0.5:
                    s = smoothstep(d, 0, 1)
                    if d > 0:
                         shadow_angle -= np.arctan(elevation/d)
                    render_tex[h][w] = interpolate_colors([object_descriptor[1], object_descriptor[2], object_descriptor[3], 255], multiply(render_tex[h][w], (abs(125*shadow_angle/np.pi)+130)/255), s)

                if d > 1:
                    shadow_angle -= np.arctan(elevation/d)
                    render_tex[h][w] = multiply(render_tex[h][w], (abs(125*shadow_angle/np.pi)+130)/255)

# %%
# Driver code
# ----------------------------------------------

# Setup texture maps
TEX_WIDTH, TEX_HEIGHT = 1920, 1080

render_tex = generate_render_texture(TEX_WIDTH, TEX_HEIGHT)
dist_tex = np.full((TEX_HEIGHT, TEX_WIDTH), fill_value=np.inf, dtype=np.float64)
shadow_map = np.zeros((TEX_HEIGHT, TEX_WIDTH), dtype=np.float64)

# Setup objects
N_OBJECTS = 3

object_descriptors = np.zeros((N_OBJECTS, 4), np.int32)

# Universal Object Description: [type, r, g, b], to be extended...
object_descriptors[0] = [RECT, 79, 77, 231]
# object_descriptors[0] = [RECT, 255, 255, 255]
# object_descriptors[0] = [LINE, 188, 136, 241]
object_descriptors[1] = [CIRCLE, 237, 173, 74]
# object_descriptors[1] = [CIRCLE, 255, 255, 255]
object_descriptors[2] = [POLYGON, 255, 0, 0]
# object_descriptors[0] = [POLYGON, 255, 255, 255]

start = time.time()
render(render_tex, TEX_WIDTH, TEX_HEIGHT, object_descriptors)
print("Took:", (time.time() - start))

# Show results
plt.axis("off")
plt.tight_layout()
plt.imshow(render_tex)
plt.show()

save_tex(render_tex)
show_image(render_tex)

# %%

# %%
