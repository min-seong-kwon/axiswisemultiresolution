# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:46:26 2022

@author: alienware
"""
import os
import numpy as np
import re
from tqdm import tqdm

import open3d as o3d


scene_name = 'Armadillo'
#meshdir = './Meshes_%s' % (scene_name)
meshdir = '../_meshes/axisres'

ori_mesh = o3d.geometry.TriangleMesh()
rec_mesh = o3d.geometry.TriangleMesh()
for ori, final_mesh in zip([False, True], [rec_mesh, ori_mesh]):
    mesh_list = os.listdir(meshdir)
    if ori:
        # continue
        mesh_list = [m for m in mesh_list if 'ori' in m]
        if len(mesh_list)==0:
            continue
    else:
        mesh_list = [m for m in mesh_list if 'ori' not in m]
        # mesh_list = [m for m in mesh_list if 'awmr' in m]
    all_vertices = []
    all_faces = np.zeros([0,4])
    
    for m_file in tqdm(mesh_list):
        vunit_x, vunit_y, vunit_z = re.findall(r'\d+', 
                                            os.path.split(m_file)[-1])[-3:]
        key = (int(vunit_x), int(vunit_y), int(vunit_z))
        
        # if not key_is_in(key, (22,25,7), (26,27,10)):
        #     continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(meshdir,m_file))
                
        if ori:
            mesh.paint_uniform_color([1,1,1])
        else:
            randomcolor = np.random.rand(3)
            mesh.paint_uniform_color(randomcolor)
        final_mesh += mesh
    
    final_mesh.remove_duplicated_vertices()
    
    print("...generating mesh...")
    # o3d.io.write_triangle_mesh(f"{scene_name}-debug_{ori}.ply",
    #                           final_mesh)
    o3d.io.write_triangle_mesh(f"{scene_name}-seammesh-{ori}.ply",
                              final_mesh)
    print("mesh generated!")                                        
