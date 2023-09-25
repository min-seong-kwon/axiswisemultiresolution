import os
import numpy as np
from tqdm import tqdm
import pickle

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit
from source.AWMR_utils import pad_awmr_from_original,make_mesh, mesh_whole_block_singularize, split_until_thres_octree, pad_3d, get_condition
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO
from utils.TSDFDataset import TSDFDataset
import igl
import open3d as o3d

from utils.evalUtils import key_is_in
import argparse

##########################################################
parser = argparse.ArgumentParser(description='Given RGBD dataset and GT(finest resolution) TSDF data, generates axiswise multiresolution TSDF volume units')
parser.add_argument('dataset_name', type=str, metavar='DATASET_NAME',
                    help='Name of the 3D dataset instance')
parser.add_argument('finest_voxel_size', type=float, metavar='FINEST_VOXEL_SIZE',
                    help='Finest Voxel Size')
parser.add_argument('thres', type=float, metavar='THRES',
                    help='Minimum geometric error to stop splitting')
parser.add_argument('sample_option', type=str, metavar='sample_option',
                    help='downsample option: pool, mean, weighted average')

parser.add_argument('-d', '--debug', dest='debug', default=False, action='store_true',
                    help='whether in debug mode. default is False')
parser.add_argument('--p1', nargs=3, type=float, dest='debug_p1', metavar='x',
                    help='the 3d coordinatate of a corner of debugging volume(cuboid)') # TODO
parser.add_argument('--p2', nargs=3, type=float, dest='debug_p2', metavar='x',
                    help='the 3d coordinatate of the opposite corner of debugging volume(cuboid)')

parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='whether to compare with GT and calculate geometric accuracy. \nDefault is False')
parser.add_argument('-v', '--version', dest='ver', default=2,
                    help='Chamfer Distance measure version.')
args = parser.parse_args()
# BUG: evaluate False로 해야 메시 구멍 안생김 (09.15)
##########################################################
if __name__=='__main__':
    dataset_name = args.dataset_name
    assert dataset_name in ['armadillo', 'thai', 'dragon'], "invalid dataset"

    finest_voxel_size = args.finest_voxel_size
    scale_factor = 0.002/finest_voxel_size
    thres = args.thres
    sample_option = args.sample_option
    assert sample_option in ['pool', 'mean', 'weighted'], "choose down sample option properly"
    evaluate = args.evaluate
    print(f"[octree] [voxsize = {finest_voxel_size}] [thres = {thres}] [option: {sample_option}]")
##########################################################
    volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.6f}/volume_origin_{finest_voxel_size:.6f}.npy')
    if args.debug:
        assert (args.debug_p1 is not None) and (args.debug_p2 is not None),\
            "when using DEBUG flag both debug_p1 and debug_p2 arguments must have exactly 3 values: x, y, z coordinates. eg. --debug_p1 -0.023710 0.421007 0.902411 --debug_p2 1.415677 1.076951 0.901995"
        p1 = np.array(args.debug_p1)
        p2 = np.array(args.debug_p2)
        vunit_start = np.floor((p1 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
        vunit_end = np.floor((p2 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
        print("------ DEBUG MODE ------")
        print(f"only processing volume units including from {p1} to {p2}...")
        print(f"i.e. from volume unit {vunit_start} to volume unit {vunit_end}")
##########################################################
    target_path = fr'../results/[TSDF]{dataset_name}/octree/voxsize_{finest_voxel_size:.6f}'
    mesh_filename = target_path + fr"/{dataset_name}_octree_thres={thres}_{sample_option}.ply"
    blockmesh_path = f'../_meshes/{dataset_name}/axisres' # for debug

    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    if not os.path.exists(blockmesh_path):
        os.makedirs(blockmesh_path, exist_ok=True)

##########################################################
    dataset_32 = TSDFDataset(dataset_name=dataset_name,
                            volume_unit_dim=32,
                            finest_voxel_size=finest_voxel_size,
                            volume_origin=volume_origin,
                            ETRI=False)

    dataset_16 = TSDFDataset(dataset_name=dataset_name,
                            volume_unit_dim=16,
                            finest_voxel_size=finest_voxel_size,
                            volume_origin=volume_origin,
                            ETRI=False)

    dataset_8 = TSDFDataset(dataset_name=dataset_name,
                            volume_unit_dim=8,
                            finest_voxel_size=finest_voxel_size,
                            volume_origin=volume_origin,
                            ETRI=False)

    volume_units_32 = {}
    for k in dataset_32.tsdf_blocks.keys():
        volume_units_32[k] = VolumeUnit(volume_unit_dim=32)
        volume_units_32[k].D = dataset_32.tsdf_blocks[k]

    volume_units_16 = {}
    for k in dataset_16.tsdf_blocks.keys():
        volume_units_16[k] = VolumeUnit(volume_unit_dim=16)
        volume_units_16[k].D = dataset_16.tsdf_blocks[k]

    volume_units_8 = {}
    for k in dataset_8.tsdf_blocks.keys():
        volume_units_8[k] = VolumeUnit(volume_unit_dim=8)
        volume_units_8[k].D = dataset_8.tsdf_blocks[k]
##########################################################
    print("splitting tsdf blocks...")

    awmr_tsdfs = {}
    for k in tqdm(volume_units_32.keys(), leave=True):
        if args.debug and not key_is_in(k, vunit_start, vunit_end): # DEBUG
                    continue
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (dataset_name, k[0], k[1], k[2])
        
        contd_condition = get_condition(volume_units_32,volume_units_16,volume_units_8,k)

        if contd_condition:
            continue
        
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8),
                                    unit_index=k,
                                    tsdf=volume_units_8[k].D)
        split_until_thres_octree(awmr_tsdfs[k], 
                        thres,
                        volume_units_16=volume_units_32,
                        volume_units_8=volume_units_16,
                        volume_units_4=volume_units_8,
                        axisres=np.array((8,8,8)),  # edit: 4x4x4 -> 8x8x8
                        unit_index=k, 
                        start_point=np.array((0,0,0)),
                        max_res=32,
                        for_train=True,
                        version=args.ver,
                        sample_option=sample_option)

##########################################################
    print("meshing...")
    counter = 0
    counter2 = 0
    unnecessary_keys = []
    wrong_keys = []

    mesh = o3d.geometry.TriangleMesh()
    for k in tqdm(awmr_tsdfs.keys(), leave=True):
        block_mesh = mesh_whole_block_singularize(awmr_tsdfs[k],
                                                unit_index=k,
                                                awmr_dict=awmr_tsdfs,
                                                node=None,
                                                volume_origin=volume_origin,
                                                voxel_size=base_voxel_size,
                                                volume_unit_dim=base_volume_unit_dim,
                                                baseres=8)
        block_mesh.scale(1/scale_factor, center=tuple(volume_origin))
        if not block_mesh.has_vertex_colors:
            block_mesh.paint_uniform_color([1, 1, 1])
        if np.sum(np.asarray(block_mesh.triangles)) == 0:
            continue
        
        if evaluate:
            print("evaulate mode on")
            ori_mesh = make_mesh(volume_units_32[k].D,
                                (base_volume_unit_dim,
                                base_volume_unit_dim,
                                base_volume_unit_dim),
                                voxel_size=base_voxel_size,
                                volume_unit_dim=base_volume_unit_dim,
                                k=k,
                                volume_origin=volume_origin,
                                allow_degenerate=False)
            ori_mesh.scale(1/scale_factor,center=tuple(volume_origin))
            ori_mesh.paint_uniform_color([1, 1, 1])
            if np.sum(np.asarray(block_mesh.triangles)) == 0:
                if np.sum(np.asarray(ori_mesh.triangles)) == 0:
                    unnecessary_keys.append(k)
                    counter2 += 1
                else:
                    wrong_keys.append(k)
                    counter += 1
                continue
            if np.sum(np.asarray(ori_mesh.triangles)) == 0:
                continue
            block_mesh.paint_uniform_color([1, 1, 1])

        mesh += block_mesh
        title = os.path.join(blockmesh_path,  f'{k[-3]}_{k[-2]}_{k[-1]}.ply')
        o3d.io.write_triangle_mesh(
            title, block_mesh, write_ascii=True, write_vertex_colors=True)

    o3d.io.write_triangle_mesh(mesh_filename,
                                mesh, write_ascii=True, write_vertex_colors=True)
