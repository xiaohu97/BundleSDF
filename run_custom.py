# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from bundlesdf import *
import argparse
import os,sys,time
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter


def format_seconds(seconds):
  seconds = max(0, int(round(seconds)))
  mins, sec = divmod(seconds, 60)
  hours, mins = divmod(mins, 60)
  if hours > 0:
    return f"{hours:d}:{mins:02d}:{sec:02d}"
  return f"{mins:02d}:{sec:02d}"


class TerminalProgressBar:
  def __init__(self, total, label='run_video', width=28):
    self.total = max(1, int(total))
    self.label = label
    self.width = width
    self.start_time = time.time()
    self.closed = False
    self.last_len = 0

  def update(self, current, frame_id=None):
    if self.closed:
      return
    current = max(0, min(int(current), self.total))
    frac = current / self.total
    filled = int(round(self.width * frac))
    bar = '#' * filled + '.' * (self.width - filled)
    elapsed = time.time() - self.start_time
    eta = 0 if current == 0 else elapsed / current * (self.total - current)
    msg = f"\r[{self.label}] [{bar}] {current}/{self.total} ({frac*100:5.1f}%)"
    if frame_id is not None:
      msg += f" frame={frame_id}"
    msg += f" elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
    padding = max(0, self.last_len - len(msg))
    sys.stdout.write(msg + (' ' * padding))
    sys.stdout.flush()
    self.last_len = len(msg)
    if current >= self.total:
      self.close()

  def close(self):
    if self.closed:
      return
    sys.stdout.write('\n')
    sys.stdout.flush()
    self.closed = True


def apply_global_refine_profile(cfg_nerf, profile='memory'):
  profile = profile.strip().lower()

  if profile == 'memory':
    cfg_nerf['n_step'] = 1000
    cfg_nerf['N_samples'] = 64
    cfg_nerf['N_samples_around_depth'] = 128
    cfg_nerf['first_frame_weight'] = 1
    cfg_nerf['down_scale_ratio'] = 1
    cfg_nerf['finest_res'] = 192
    cfg_nerf['num_levels'] = 12
    cfg_nerf['mesh_resolution'] = 0.004
    cfg_nerf['n_train_image'] = 120
    cfg_nerf['fs_sdf'] = 0.1
    cfg_nerf['frame_features'] = 2
    cfg_nerf['rgb_weight'] = 100
    tex_res = 256
    reader_kwargs = {'shorter_side': 480}
  elif profile == 'full':
    cfg_nerf['n_step'] = 2000
    cfg_nerf['N_samples'] = 64
    cfg_nerf['N_samples_around_depth'] = 256
    cfg_nerf['first_frame_weight'] = 1
    cfg_nerf['down_scale_ratio'] = 1
    cfg_nerf['finest_res'] = 256
    cfg_nerf['num_levels'] = 16
    cfg_nerf['mesh_resolution'] = 0.002
    cfg_nerf['n_train_image'] = 500
    cfg_nerf['fs_sdf'] = 0.1
    cfg_nerf['frame_features'] = 2
    cfg_nerf['rgb_weight'] = 100
    tex_res = 512
    reader_kwargs = {'downscale': 1}
  else:
    raise ValueError(f"Unsupported global refine profile: {profile}")

  cfg_nerf['i_img'] = np.inf
  cfg_nerf['i_mesh'] = cfg_nerf['i_img']
  cfg_nerf['i_nerf_normals'] = cfg_nerf['i_img']
  cfg_nerf['i_save_ray'] = cfg_nerf['i_img']
  return tex_res, reader_kwargs


def run_one_video(video_dir='/home/bowen/debug/2022-11-18-15-10-24_milk', out_folder='/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk/', use_segmenter=False, use_gui=False, auto_global_refine=False, global_refine_profile='memory', erode_mask=1):
  set_seed(0)

  if os.path.isfile(video_dir) and video_dir.endswith('.bag'):
    raise RuntimeError(
      "run_custom.py expects --video_dir to be an extracted dataset folder, not a .bag file.\n"
      "Please first run:\n"
      f"  python {code_dir}/convert_realsense_bag.py --bag {video_dir}\n"
      "Then pass the generated folder path to --video_dir."
    )

  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(args.debug_level)
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = int(erode_mask)
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5
  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  if use_segmenter:
    segmenter = Segmenter()

  tracker = None
  progress = None
  try:
    tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=use_gui)

    reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)
    frame_indices = list(range(0, len(reader.color_files), args.stride))
    progress = TerminalProgressBar(total=len(frame_indices), label='run_video')

    for step_idx, i in enumerate(frame_indices, start=1):
      color_file = reader.color_files[i]
      color = cv2.imread(color_file)
      H0, W0 = color.shape[:2]
      depth = reader.get_depth(i)
      H,W = depth.shape[:2]
      color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
      depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)

      if i==0:
        mask = reader.get_mask(0)
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
        if use_segmenter:
          mask = segmenter.run(color_file.replace('rgb','masks'))
      else:
        if use_segmenter:
          mask = segmenter.run(color_file.replace('rgb','masks'))
        else:
          mask = reader.get_mask(i)
          mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

      if cfg_bundletrack['erode_mask']>0:
        kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel)

      id_str = reader.id_strs[i]
      pose_in_model = np.eye(4)

      K = reader.K.copy()

      mask_pixels = int((mask > 0).sum())
      valid_depth_pixels = int(((depth >= 0.1) & (mask > 0)).sum())
      if mask_pixels == 0:
        print(f"[run_custom.py] Warning: frame {id_str} mask empty after segmentation/erosion")
      elif valid_depth_pixels == 0:
        print(f"[run_custom.py] Warning: frame {id_str} mask has {mask_pixels} foreground pixels but zero valid depth")

      tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)
      progress.update(step_idx, frame_id=id_str)
  finally:
    if progress is not None:
      progress.close()
    if tracker is not None:
      tracker.on_finish()

  if auto_global_refine:
    print(f"run_video finished. Starting global_refine with profile={global_refine_profile}")
    run_one_video_global_nerf(out_folder=out_folder, global_refine_profile=global_refine_profile)
  else:
    print("run_video finished. Skipping automatic global_refine.")
    print("Run it manually when needed:")
    print(f"  python {code_dir}/run_custom.py --mode global_refine --video_dir {video_dir} --out_folder {out_folder}")



def run_one_video_global_nerf(out_folder='/home/bowen/debug/bundlesdf_scan_coffee_415', global_refine_profile='memory'):
  set_seed(0)

  out_folder += '/'   #!NOTE there has to be a / in the end

  cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml",'r'))
  cfg_bundletrack['debug_dir'] = out_folder
  cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml",'r'))
  tex_res, reader_kwargs = apply_global_refine_profile(cfg_nerf, profile=global_refine_profile)

  cfg_nerf['datadir'] = f"{out_folder}/nerf_with_bundletrack_online"
  cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])

  os.makedirs(cfg_nerf['datadir'],exist_ok=True)

  cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  reader = YcbineoatReader(video_dir=args.video_dir, **reader_kwargs)

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, tracking_mode=False)
  tracker.cfg_nerf = cfg_nerf
  tracker.run_global_nerf(reader=reader, get_texture=True, tex_res=tex_res)
  tracker.on_finish()

  print(f"Done")


def postprocess_mesh(out_folder):
  mesh_files = sorted(glob.glob(f'{out_folder}/**/nerf/*normalized_space.obj',recursive=True))
  print(f"Using {mesh_files[-1]}")
  os.makedirs(f"{out_folder}/mesh/",exist_ok=True)

  print(f"\nSaving meshes to {out_folder}/mesh/\n")

  mesh = trimesh.load(mesh_files[-1])
  with open(f'{os.path.dirname(mesh_files[-1])}/config.yml','r') as ff:
    cfg = yaml.load(ff)
  tf = np.eye(4)
  tf[:3,3] = cfg['translation']
  tf1 = np.eye(4)
  tf1[:3,:3] *= cfg['sc_factor']
  tf = tf1@tf
  mesh.apply_transform(np.linalg.inv(tf))
  mesh.export(f"{out_folder}/mesh/mesh_real_scale.obj")

  components = trimesh_split(mesh, min_edge=1000)
  best_component = None
  best_size = 0
  for component in components:
    dists = np.linalg.norm(component.vertices,axis=-1)
    if len(component.vertices)>best_size:
      best_size = len(component.vertices)
      best_component = component
  mesh = trimesh_clean(best_component)

  mesh.export(f"{out_folder}/mesh/mesh_biggest_component.obj")
  mesh = trimesh.smoothing.filter_laplacian(mesh,lamb=0.5, iterations=3, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)
  mesh.export(f'{out_folder}/mesh/mesh_biggest_component_smoothed.obj')



def draw_pose():
  K = np.loadtxt(f'{args.out_folder}/cam_K.txt').reshape(3,3)
  color_files = sorted(glob.glob(f'{args.out_folder}/color/*'))
  mesh = trimesh.load(f'{args.out_folder}/textured_mesh.obj')
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  out_dir = f'{args.out_folder}/pose_vis'
  os.makedirs(out_dir, exist_ok=True)
  logging.info(f"Saving to {out_dir}")
  for color_file in color_files:
    color = imageio.imread(color_file)
    pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
    pose = pose@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
    id_str = os.path.basename(color_file).replace('.png','')
    imageio.imwrite(f'{out_dir}/{id_str}.png', vis)



if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default="run_video", help="run_video/global_refine/draw_pose")
  parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
  parser.add_argument('--out_folder', type=str, default="/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk")
  parser.add_argument('--use_segmenter', type=int, default=0)
  parser.add_argument('--use_gui', type=int, default=1)
  parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
  parser.add_argument('--debug_level', type=int, default=2, help='higher means more logging')
  parser.add_argument('--auto_global_refine', type=int, default=0, help='run global_refine automatically after run_video; default 0 to avoid surprise memory spikes')
  parser.add_argument('--global_refine_profile', type=str, default='memory', choices=['memory', 'full'], help='memory: lower-memory defaults; full: original heavy settings')
  parser.add_argument('--erode_mask', type=int, default=1, help='mask erosion kernel size for run_video; default 1 is more stable than the old hardcoded 3')
  args = parser.parse_args()

  if args.mode=='run_video':
    run_one_video(video_dir=args.video_dir, out_folder=args.out_folder, use_segmenter=args.use_segmenter, use_gui=args.use_gui, auto_global_refine=bool(args.auto_global_refine), global_refine_profile=args.global_refine_profile, erode_mask=args.erode_mask)
  elif args.mode=='global_refine':
    run_one_video_global_nerf(out_folder=args.out_folder, global_refine_profile=args.global_refine_profile)
  elif args.mode=='draw_pose':
    draw_pose()
  else:
    raise RuntimeError
