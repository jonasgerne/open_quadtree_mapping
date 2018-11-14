import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyQuadtreeMapping
import random

def load_pfm(filename):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(filename, errors='ignore') as file:
        header = file.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape), scale
    return None

def export_binary_ply(output_path, vertices, colors=None, normals=None):
    from plyfile import PlyData, PlyElement

    if colors is None:
        vertices = [tuple(x) for x in vertices.tolist()]
        vertices = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertices, "vertex")
    elif normals is None:
        vertices = np.concatenate((vertices, colors), axis=1)
        vertices = [tuple(x) for x in vertices.tolist()]
        vertices = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                             ('blue', 'u1')])
        el = PlyElement.describe(vertices, "vertex")
    else:
        vertices = np.concatenate((vertices, normals, colors), axis=1)
        vertices = [tuple(x) for x in vertices.tolist()]
        vertices = np.array(vertices,
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'),
                                   ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertices, "vertex")
    PlyData([el]).write(output_path)


if __name__ == "__main__":
    base_dir = "F:/_DATASETS/KITTI_Raw/2011_09_26/2011_09_26_drive_0095_sync/image_02"
    out_dir = "F:/_DATASETS/KITTI_Raw/2011_09_26_drive_0095_sync"
    posesPath = "poses.txt"
    N = 2 # 5
    wait = True
    b = 0.5327254400790535
    scale = 1.0
    minDepth = 1.0
    maxDepth = 50.0

    cost_downsampling = 1
    doBeliefPropagation = True
    useQuadtree = False
    P1 = 0.003
    P2 = 0.01
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    poses = np.loadtxt(posesPath).reshape((-1,3,4)).astype(np.float32)
    poses = np.concatenate((poses, np.tile(np.eye(4,4,dtype=poses.dtype)[np.newaxis,3:4,:], (poses.shape[0], 1, 1))), axis=1)
   
    K = np.loadtxt(os.path.join(base_dir, "K.txt")).reshape((3,3)).astype(np.float32)
    if scale != 1.0:
        K[:2] *= scale

    cm = plt.get_cmap('jet')

    height, width = cv2.imread(os.path.join(base_dir, "data", "{:010d}.png".format(0)), cv2.IMREAD_GRAYSCALE).shape
    if scale != 1.0:
        height = int(scale * height)
        width = int(scale * width)

    pyQuadtreeMapping.initialize(K, width, height, cost_downsampling, doBeliefPropagation, useQuadtree, P1, P2)

    Xv, Yv = np.meshgrid(np.arange(width), np.arange(height))
    Xtemp = np.concatenate((Xv[:, :, np.newaxis].astype(np.float32), Yv[:, :, np.newaxis].astype(np.float32)), axis=-1)

    # Back projection
    x_raw = np.concatenate((Xtemp, np.ones((height, width, 1), dtype=np.float)), axis=-1)
    X_raw = np.squeeze(np.matmul(np.linalg.inv(K), np.expand_dims(x_raw, -1)))

    #for idx in range(N-1, poses.shape[0]-1):
    for idx in range(poses.shape[0]-N):
        #if idx == N-1:
        #    T = poses[idx::-1].copy()
        #else:
        #    T = poses[idx:idx-N:-1].copy()
        T = poses[idx:idx+N].copy()

        # Normalize poses
        T = np.matmul(np.linalg.inv(T[0]), T)

        pD = np.zeros((0,),dtype=np.float32)
        pN = np.zeros((0,),dtype=np.float32)

        # RGB
        I = np.zeros((N, height, width, 1), dtype=np.uint8)
        for i in range(N):
            #img = np.expand_dims(cv2.imread(os.path.join(base_dir, "data", "{:010d}.png".format(idx - i)), cv2.IMREAD_GRAYSCALE).astype(np.float32), -1)
            img = np.expand_dims(cv2.imread(os.path.join(base_dir, "data", "{:010d}.png".format(idx + i)), cv2.IMREAD_GRAYSCALE).astype(np.float32), -1)
            img = np.expand_dims(cv2.resize(img, (width, height), None, interpolation=cv2.INTER_LINEAR), -1)
            I[i] = img

        cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("rgb", int(width / scale), int(height / scale))
        cv2.imshow("rgb", I[0])

        depth, = pyQuadtreeMapping.compute(I, T)

        depth_norm = depth.copy()
        depth_color = cm(1.0 - depth_norm/maxDepth)
        depth_color[depth_norm < 0] = np.nan
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", int(width / scale), int(height / scale))
        cv2.imshow("depth", depth_color)

        if wait:
            cv2.waitKey()
        else:
            cv2.waitKey(100)

        verts = np.multiply(X_raw,np.tile(np.expand_dims(depth, -1), (1,1,3))).astype(np.float32)
        colors = cv2.cvtColor(I[0], cv2.COLOR_GRAY2RGB)

        mask = depth > 0
        
        verts = verts[mask]
        colors = colors[mask]

        export_binary_ply("pointcloud_{}.ply".format(idx), verts.reshape((-1, 3)), colors.reshape((-1,3)))

        print("Processed {} of {} files".format(idx, poses.shape[0]-1))

