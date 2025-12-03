'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-08-09 15:56:09
Description: 
         ▄              ▄
        ▌▒█           ▄▀▒▌     
        ▌▒▒▀▄       ▄▀▒▒▒▐
       ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
     ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
   ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
  ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
  ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
  ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
    ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
      ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
         ▒▒▒▒▒▒▒▒▒▒▀▀
When I wrote this, only God and I understood what I was doing
Now, God only knows
'''
import numpy as np
from PIL import Image
from os.path import *
import re
import imageio
import cv2
import os
import Imath,OpenEXR
import array
from collections import defaultdict
# from conversion_tools.exr_processing.color_convertion.colorutil import Color_transform
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
HALF  = Imath.PixelType(Imath.PixelType.HALF)
UINT  = Imath.PixelType(Imath.PixelType.UINT)

NO_COMPRESSION    = Imath.Compression(Imath.Compression.NO_COMPRESSION)
RLE_COMPRESSION   = Imath.Compression(Imath.Compression.RLE_COMPRESSION)
ZIPS_COMPRESSION  = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
ZIP_COMPRESSION   = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
PIZ_COMPRESSION   = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
PXR24_COMPRESSION = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)

transformer = None

NP_PRECISION = {
  "FLOAT": np.float32,
  "HALF":  np.float16,
  "UINT":  np.uint32
}
_default_channel_names = {
  1: ['Z'],
  2: ['X','Y'],
  3: ['R','G','B'],
  4: ['R','G','B','A']
}
TAG_CHAR = np.array([202021.25], np.float32)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def prune(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() not in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() not in x.lower(),c))
    return res 
def gofind(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() in x.lower(),c)) 
    return res  
def custom_refine(flow,zero_to_one=True):
    height,width = flow[...,0].shape
    #average value
    # if zero_to_one:
    #     flow[...,0]/=width
    #     flow[...,1]/=-height
    # else:
    #     flow[...,0]/=-width
    #     flow[...,1]/=height
    flow[...,0]/=width
    flow[...,1]/=height

    # if flow.shape[2] >= 3:
    #     flow[...,2] /= 65535 #65535*255
    return flow
    
    
def write_flo_file(flow, filename): # flow: H x W x 2
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    if flow.ndim == 4: # has batch
        flow = flow[0]

    outpath = os.path.dirname(filename)
    if outpath != '' and not os.path.isdir(outpath):
        os.makedirs(outpath)

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w, h = np.fromfile(f, np.int32, count=2)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:       # little-endian
        endian = '<'
        scale = -scale
    else:               # big-endian
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def readFlowNpz(filename):
    npz = np.load(filename)
    u = npz['u']
    v = npz['v']

    flow = np.stack((u, v), -1)
    valid = np.all(np.isfinite(flow), axis=-1)

    return flow, valid


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext =='.exr':
        flow = imageio.imread(file_name)[...,:2].astype(np.float32)
        h,w,_ = flow.shape
        flow[...,0] *= w
        flow[...,1] *= h
        return flow
    return []


def read_flow_sparse(filename):
    ext = splitext(filename)[-1]

    if ext == '.png':
        return readFlowKITTI(filename)
    elif ext == '.npz':
        return readFlowNpz(filename)
    elif ext == '.flo':
        flow = read_flo_file(filename).astype(np.float32)
        return flow, np.all(np.abs(flow) <= 1e9, axis=-1)
    elif ext =='.exr':
        flow = imageio.imread(filename).astype(np.float32)
        h,w,_ = flow.shape
        flow[...,0] *= w
        flow[...,1] *= h
        return flow[...,:2],flow[...,-1] ==1
    else:
        raise ValueError("unsupported flow file format")


def exr_imread(filePath,pt=FLOAT):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    if 'A' in img_exr.header()['channels']:
        r_str, g_str, b_str ,a_str= img_exr.channels('RGBA',pt)
        red = np.array(array.array('f', r_str))
        green = np.array(array.array('f', g_str))
        blue = np.array(array.array('f', b_str))
        alpha = np.array(array.array('f', a_str))
    else:
        r_str, g_str, b_str = img_exr.channels('RGB',pt)
        red = np.array(array.array('f', r_str))
        green = np.array(array.array('f', g_str))
        blue = np.array(array.array('f', b_str))
        alpha = np.zeros_like(red)
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    alpha = alpha.reshape(size)
    image = np.stack([red,green,blue,alpha],axis=2)
    return image.astype('float32')

def mvread(path):
    try:
      res = imageio.imread(path).astype('float32')
    except:
      raise EOFError('[MM ERROR][file]The file is damaged. Please clean the cache and run again')
    return res
def mvread2(path):
    try:
      res = exr_imread(path)
    except:
      raise EOFError('[MM ERROR][file]The file is damaged. Please clean the cache and run again')
    return res

def read(path,type='flo',lut_file=None,self_mask=False,OPENEXR=True,Unrealmode=False,color_space=None):
    mvr = mvread3 if OPENEXR else mvread
    if path is None or path.lower()=='none' or not os.path.isfile(path):
        return None
    res = None
    if type.lower() == 'flo':
        if '.exr' in path:
            res =  mvr(path)
        elif '.bin' in path or '.raw' in path:
            return np.load(path).astype(np.float32)
        elif '.pfm' in path:
            return readPFM(path).astype(np.float32)
        elif '.flo' in path:
            res =  read_flo_file(path).astype('float32')

    if type.lower() == 'mask':
        if self_mask:
            if '.exr' in path:
                tmp = mvr(path)
            else:
                tmp = cv2.imread(path)[...,::-1].astype('float32')
            res = np.zeros_like(tmp)
            res[np.where(tmp>0)] = 1
        else:
            if '.exr' in path:
                res = mvr(path)
            else:
                res = cv2.imread(path)
                if len(res.shape) == 2:
                    res = res[...,None]
                res = res[...,::-1].astype('float32')
        if res.max()>255:
            res/=65535
        elif res.max()>1:
            res/=255
        res = np.clip(res,0,1)
        res = res[...,-1] if not Unrealmode else res[...,0]

    if type.lower() == 'image' or type.lower() == 'ldr':
        if lut_file is not None and os.path.isfile(lut_file):
            import colour
            global LUT
            if LUT is None:
                LUT = colour.read_LUT(lut_file)
            assert '.exr' in path
            tmp = mvr(path)
            res = LUT.apply(tmp,interpolator=colour.algebra.table_interpolation_trilinear)*255
        else:
            if '.exr' in path:
              res = mvr(path)
            elif '.tif' in path:
              res = cv2.imread(path,-1)[...,::-1]/65535
            else:
              res = cv2.imread(path)[...,::-1]/255
        if type.lower() == 'image':
           res = (np.clip(res,0,1)*255).astype('uint8')
        res = res[...,:3]
    
    if type.lower() == 'hdr':
        res = mvr(path)[...,:3]
        if lut_file is not None and os.path.isfile(lut_file):
            import colour
            global LUT_HDR
            if LUT_HDR is None:
                LUT = colour.read_LUT(lut_file)
            res = LUT_HDR.apply(res,interpolator=colour.algebra.table_interpolation_trilinear)
    
    if type.lower() == 'gray':
        if '.exr' in path:
            tmp = np.clip(mvr(path),0,1)
            res = (tmp*255)[...,:3].astype('uint8')
        else:
            res =  cv2.imread(path).astype('uint8')[...,::-1]
        res = res[...,0]
        
    return np.ascontiguousarray(res)



def pickle_write(path,data):
    import pickle
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)

def pickle_read(path):
    import pickle
    pickled_dat = open (path, "rb")
    return pickle.load(pickled_dat)

def mvwrite1(path,flow,compress='none',precision = FLOAT):
    if compress.lower()=='none':
        imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE) 
    else:
        imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT) 


def mvwrite_helper(filename, data, channel_names = None, precision = FLOAT, compression = PIZ_COMPRESSION, extra_headers={}):
  # Helper function add a third dimension to 2-dimensional matrices (single channel)
  def make_ndims_3(matrix):
    if matrix.ndim > 3 or matrix.ndim < 2:
      raise Exception("Invalid number of dimensions for the `matrix` argument.")
    elif matrix.ndim == 2:
      matrix = np.expand_dims(matrix, -1)
    return matrix

  # Helper function to read channel names from default
  def get_channel_names(channel_names, depth):
    if channel_names:
      if depth != len(channel_names):
        raise Exception("The provided channel names have the wrong length (%d vs %d)." % (len(channel_names), depth))
      return channel_names
    elif depth in _default_channel_names:
      return _default_channel_names[depth]
    else:
      raise Exception("There are no suitable default channel names for data of depth %d" % depth)

  #
  # Case 1, the `data` argument is a dictionary
  #
  if isinstance(data, dict):
    # Make sure everything has ndims 3
    for group, matrix in data.items():
      data[group] = make_ndims_3(matrix)

    # Prepare precisions
    if not isinstance(precision, dict):
      precisions = {group: precision for group in data.keys()}
    else:
      precisions = {group: precision.get(group, FLOAT) for group in data.keys()}

    # Prepare channel names
    if channel_names is None:
      channel_names = {}
    channel_names = {group: get_channel_names(channel_names.get(group), matrix.shape[2]) for group, matrix in data.items()}

    # Collect channels
    channels = {}
    channel_data = {}
    width = None
    height = None
    for group, matrix in data.items():
      # Read the depth of the current group
      # and set height and width variables if not set yet
      if width is None:
        height, width, depth = matrix.shape
      else:
        depth = matrix.shape[2]
      names = channel_names[group]
      # Check the number of channel names
      if len(names) != depth:
        raise Exception("Depth does not match the number of channel names for channel '%s'" % group)
      for i, c in enumerate(names):
        if group == "default":
          channel_name = c
        else:
          channel_name = "%s.%s" % (group, c)
        channels[channel_name] = Imath.Channel(precisions[group])
        channel_data[channel_name] = matrix[:,:,i].astype(NP_PRECISION[str(precisions[group])]).tobytes()

    # Save
    header = OpenEXR.Header(width, height)
    if extra_headers:
      header = dict(header, **extra_headers)
    header['compression'] = compression
    header['channels'] = channels
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels(channel_data)

  #
  # Case 2, the `data` argument is one matrix
  #
  elif isinstance(data, np.ndarray):
    data = make_ndims_3(data)
    height, width, depth = data.shape
    channel_names = get_channel_names(channel_names, depth)
    header = OpenEXR.Header(width, height)
    if extra_headers:
      header = dict(header, **extra_headers)
    header['compression'] = compression
    header['channels'] = {c: Imath.Channel(precision) for c in channel_names}
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({c: data[:,:,i].astype(NP_PRECISION[str(precision)]).tobytes() for i, c in enumerate(channel_names)})

  else:
    raise Exception("Invalid precision for the `data` argument. Supported are NumPy arrays and dictionaries.")

def mvwrite2(path,flow,compress='piz',precision = FLOAT):
    if compress.lower() == 'rle':
        cpm = RLE_COMPRESSION
    elif compress.lower() == 'zips':
        cpm = ZIPS_COMPRESSION
    elif compress.lower() == 'zip':
        cpm = ZIP_COMPRESSION
    elif compress.lower() == 'piz':
        cpm = PIZ_COMPRESSION
    elif compress.lower() == 'pxr':
        cpm = PXR24_COMPRESSION
    else:
        cpm = NO_COMPRESSION
    mvwrite_helper(path, flow, precision = precision, compression = cpm)

def mvwrite(path,flow,compress='piz',OPENEXR=True,precision = 'float'):
    if precision.lower() == 'half':
      precision_ = HALF
    elif precision.lower() == 'uint':
      precision_ = UINT
    else:
      precision_ = FLOAT
    writer = mvwrite2 if OPENEXR else mvwrite1
    if not os.path.exists(os.path.dirname(path)):  # 判断目录是否存在
        os.makedirs(os.path.dirname(path),exist_ok=True) 
    if '.exr' in path:
        if len(flow.shape) == 2:
            flow = np.repeat(flow[...,None],3,axis=2)
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        writer(path,flow,compress,precision_)
    elif '.flo' in path:
        write_flo_file(flow[...,:2],path)
    elif '.tif' in path:
        if len(flow.shape) == 2:
          flow = np.repeat(flow[...,None],3,axis=2)
        if flow.shape[2] == 2:
          flow = np.insert(flow,2,0,axis=2)
        flow = np.clip(flow,-1,1)
        flow *= 65535
        flow = flow.astype('uint16')
        if flow.shape[2] ==4:
          Image.fromarray(flow).save(path)
        else:
          cv2.imwrite(path,flow[...,:3][...,::-1])
    else:
        if len(flow.shape) == 2:
          flow = np.repeat(flow[...,None],3,axis=2)
        if flow.shape[2] == 2:
          flow = np.insert(flow,2,0,axis=2)
        flow = np.clip(flow,-1,1)
        if flow.min() < 0:
          flow = ((flow+1)/2 * 255).astype('uint8')
        else:
          flow = (flow * 255).astype('uint8')
        if flow.shape[2] ==4:
          Image.fromarray(flow).save(path)
        else:
          cv2.imwrite(path,flow[...,:3][...,::-1])

def write(path,flow,compress='piz'):
    if flow is None or path is None:
      return
    if type(path) == str:
      mvwrite(path,flow,compress)
    else:
      mvwrite(flow,path,compress)

    #front masked area set to 0.5 and back to 1 
def save_mv_file(save_name,opt,valid,args):
    if len(save_name) != len(valid) or len(save_name) != opt.shape[0]:
      raise NameError('[MM ERROR][image] Incorrect naming convention')
    for i in range(len(valid)):
        if args.pass_when_exist and save_name[i] is not None and os.path.isfile(save_name[i]):
            continue
        if save_name[i] is not None:
            if valid[i]:
                zero_to_one = i < len(valid)//2
                tmp_opt = opt[i]
                if args.edge_filter:
                   from myutil import mv_magnitude
                   tmp_opt = mv_magnitude(tmp_opt)
                save_file(save_name[i],tmp_opt,refine=args.refine,savetype=args.savetype,zero_to_one=zero_to_one,compress_method=args.compress_method.lower())
            else:
                tmp_opt = np.zeros((args.h,args.w,4)).astype('float32')
                if opt is not None and opt[i] is not None and opt[i].shape[2] == 4:
                    tmp_opt[...,-1] = opt[i][...,-1]
                save_file(save_name[i],tmp_opt,refine=args.refine,savetype=args.savetype,zero_to_one=False,compress_method=args.compress_method.lower())

def save_file(save_path,flow,refine=False,savetype='exr',zero_to_one=True,compress_method='none'):
    flow = flow.astype("float32")
    #append z axis
    if flow.shape[2] == 2:
        flow = np.insert(flow,2,0,axis=2)
    #refine
    if refine:
        flow = custom_refine(flow,zero_to_one=zero_to_one)
    #append
    if flow.shape[2] == 3 and savetype=='exr':
        flow = np.insert(flow,2,0,axis=2)
    #mkdir
    if not os.path.exists(os.path.dirname(save_path)):  # 判断目录是否存在
        os.makedirs(os.path.dirname(save_path),exist_ok=True) 
    # if savetype == 'exr':           
    mvwrite(save_path,flow,compress_method)




def save_depth_file(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    # if zero_to_one:
    #     flow = -flow
    # flow = np.abs(flow)
    # flow = (flow + dp_value) / dp_value*2
    
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    flow = (flow + depth_range/2).clip(0, depth_range) / depth_range
    h,w,_ = flow.shape
    res = np.ones((h,w,4)).astype("float32")
    if reverse:
        flow *= -1
    res[...,3] = flow[...,0]
    res[...,2] = flow[...,1]
    imageio.imwrite(save_path,res.astype("float32"),flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)


def save_depth_file_flo(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    if reverse:
        flow *= -1
    write_flo_file(flow.astype("float32"),save_path.replace('.exr','.flo'))

def save_depth_file_single(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    if reverse:
        flow *= -1
    np.savetxt(save_path.replace('.exr','.log'), flow[...,1],fmt='%f',delimiter=' ')

def color_space_check(res,args):
    if args.color_space.lower() == 'image' or args.color_space.lower() == 'ldr':
       return res
    global transformer
    src = args.color_space
    tar = 'in_rec709'
    if '-acescg' in args.algorithm.lower():
        tar = 'acescg'
    res = color_trans(res,src,tar)
    return res


def color_trans(res,src,tar):
    global transformer
    if src != tar:    
        if transformer is None:
                transformer = Color_transform(src,tar)
        transformer.apply(res)
    return res  

'''
description: 加载图片并且过滤掉mask部分
param {*} image 图片dict
param {*} mask_file mask文件dict
param {*} threshold 过滤阈值
param {*} reverse 是否翻转mask(未启用,在tiff文件中需要启用)
return {*} 图片结果
'''
def mask_read(image,mask_file,args):
    res = read(image,type=args.color_space,lut_file=args.lut_file,self_mask=args.self_mask,color_space=args.color_space)
    res = color_space_check(res,args) 
    if mask_file is not None:
        mask = read(mask_file,type='mask')
        if res.shape[:2] != mask.shape:
            raise ValueError('[MM ERROR][mask]mask size is wrong! imgsize = {},mask_size={}'.format(res.shape[:2],mask.shape))
            mask = cv2.resize(mask,(res.shape[1],res.shape[0]),interpolation=cv2.INTER_NEAREST)
        res = np.concatenate([res,mask[...,None]],axis=-1)      
    return res.astype('float32')



def exropen(filename):
  # Check if the file is an EXR file
  if not OpenEXR.isOpenExrFile(filename):
    raise Exception("File '%s' is not an EXR file." % filename)
  # Return an `InputFile`
  return InputFile(OpenEXR.InputFile(filename), filename)

def _is_list(x):
  return isinstance(x, (list, tuple, np.ndarray))

def mvread3(filename, channels = "default", precision = FLOAT):
  f = exropen(filename)
  if _is_list(channels):
    # Construct an array of precisions
    return f.get_dict(channels, precision=precision)

  else:
    return f.get(channels, precision)

def read_all(filename, precision = FLOAT):
  f = exropen(filename)
  return f.get_all(precision=precision)


class InputFile(object):

  def __init__(self, input_file, filename=None):
    self.input_file = input_file

    if not input_file.isComplete():
      raise Exception("EXR file '%s' is not ready." % filename)

    header = input_file.header()
    dw     = header['dataWindow']

    self.width             = dw.max.x - dw.min.x + 1
    self.height            = dw.max.y - dw.min.y + 1
    self.channels          = sorted(header['channels'].keys(),key=_channel_sort_key)
    self.depth             = len(self.channels)
    self.precisions        = [c.type for c in header['channels'].values()]
    self.channel_precision = {c: v.type for c, v in header['channels'].items()}
    self.channel_map       = defaultdict(list)
    self.root_channels     = set()
    self._init_channel_map()

  def _init_channel_map(self):
    # Make a dictionary of subchannels per channel
    for c in self.channels:
      self.channel_map['all'].append(c)
      parts = c.split('.')
      if len(parts) == 1:
        self.root_channels.add('default')
        self.channel_map['default'].append(c)
      else:
        self.root_channels.add(parts[0])
      for i in range(1, len(parts)+1):
        key = ".".join(parts[0:i])
        self.channel_map[key].append(c)

  def describe_channels(self):
    if 'default' in self.root_channels:
      for c in self.channel_map['default']:
        print (c)
    for group in sorted(list(self.root_channels)):
      if group != 'default':
        channels = self.channel_map[group]
        print("%-20s%s" % (group, ",".join([c[len(group)+1:] for c in channels])))

  def get(self, group = 'default', precision=FLOAT):
    channels = self.channel_map[group]

    if len(channels) == 0:
      print("I did't find any channels in group '%s'." % group)
      print("You could try:")
      self.describe_channels()
      raise Exception("I did't find any channels in group '%s'." % group)

    strings = self.input_file.channels(channels)

    matrix = np.zeros((self.height, self.width, len(channels)), dtype=NP_PRECISION[str(precision)])
    for i, string in enumerate(strings):
      precision = NP_PRECISION[str(self.channel_precision[channels[i]])]
      matrix[:,:,i] = np.frombuffer(string, dtype = precision) \
                        .reshape(self.height, self.width)
    return matrix

  def get_all(self, precision = {}):
    return self.get_dict(self.root_channels, precision)

  def get_dict(self, groups = [], precision = {}):

    if not isinstance(precision, dict):
      precision = {group: precision for group in groups}

    return_dict = {}
    todo = []
    for group in groups:
      group_chans = self.channel_map[group]
      if len(group_chans) == 0:
        print("I didn't find any channels for the requested group '%s'." % group)
        print("You could try:")
        self.describe_channels()
        raise Exception("I did't find any channels in group '%s'." % group)
      if group in precision:
        p = precision[group]
      else:
        p = FLOAT
      matrix = np.zeros((self.height, self.width, len(group_chans)), dtype=NP_PRECISION[str(p)])
      return_dict[group] = matrix
      for i, c in enumerate(group_chans):
        todo.append({'group': group, 'id': i, 'channel': c})

    if len(todo) == 0:
      print("Please ask for some channels, I cannot process empty queries.")
      print("You could try:")
      self.describe_channels()
      raise Exception("Please ask for some channels, I cannot process empty queries.")

    strings = self.input_file.channels([c['channel'] for c in todo])

    for i, item in enumerate(todo):
      precision = NP_PRECISION[str(self.channel_precision[todo[i]['channel']])]
      return_dict[item['group']][:,:,item['id']] = \
          np.frombuffer(strings[i], dtype = precision) \
            .reshape(self.height, self.width)
    return return_dict

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    self.input_file.close()



def _sort_dictionary(key):
  if key == 'R' or key == 'r':
    return "000010"
  elif key == 'G' or key == 'g':
    return "000020"
  elif key == 'B' or key == 'b':
    return "000030"
  elif key == 'A' or key == 'a':
    return "000040"
  elif key == 'X' or key == 'x':
    return "000110"
  elif key == 'Y' or key == 'y':
    return "000120"
  elif key == 'Z' or key == 'z':
    return "000130"
  else:
    return key


def _channel_sort_key(i):
  return [_sort_dictionary(x) for x in i.split(".")]


def yuv420_to_rgb(yuv_file, width, height):
  # 计算每帧的大小
  frame_size = width * height + (width // 2) * (height // 2) * 2
  
  with open(yuv_file, 'rb') as f:
      yuv_data = np.frombuffer(f.read(), dtype=np.uint8)
  
  # 确保数据长度正确
  assert len(yuv_data) % frame_size == 0, "文件大小与分辨率不匹配。"
  
  # 提取YUV平面
  y_plane = yuv_data[:width * height].reshape((height, width))
  u_plane = yuv_data[width * height:width * height + (width // 2) * (height // 2)].reshape((height // 2, width // 2))
  v_plane = yuv_data[width * height + (width // 2) * (height // 2):].reshape((height // 2, width // 2))
  
  # 将UV平面扩展到与Y平面相同的大小
  u_plane = u_plane.repeat(2, axis=0).repeat(2, axis=1)
  v_plane = v_plane.repeat(2, axis=0).repeat(2, axis=1)
  
  # 转换为RGB
  yuv = np.stack((y_plane, u_plane, v_plane), axis=-1).astype(np.float32)
  yuv[:, :, 0] -= 16
  yuv[:, :, 1:] -= 128
  
  # YUV to RGB 变换矩阵
  m = np.array([[1.164,  0.000,  1.596],
                [1.164, -0.392, -0.813],
                [1.164,  2.017,  0.000]])
  
  rgb = yuv @ m.T
  rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
  return rgb


def yuv_from_picture(filename, height, width):
    """
    param:
        filename: 待处理 YUV 图片的名字
        param height: YUV 图片的高
        param width: YUV图片的宽

    return: 
        img: 返回yv12 的数据格式
    """
    fp = open(filename, 'rb')  

    framesize = height * width * 3 // 2  # 一帧图像所含的像素个数
    uv_height = height // 2
    uv_width = width // 2

    Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
    Ut = np.zeros(shape=(uv_height, uv_width), dtype='uint8', order='C')
    Vt = np.zeros(shape=(uv_height, uv_width), dtype='uint8', order='C')

    for m in range(height):
        for n in range(width):
            Yt[m, n] = ord(fp.read(1))
    for m in range(uv_height):
        for n in range(uv_width):
            Ut[m, n] = ord(fp.read(1))
    for m in range(uv_height):
        for n in range(uv_width):
            Vt[m, n] = ord(fp.read(1))

    UV_cat = np.concatenate((Ut,Vt),axis=1)
    img = np.concatenate((Yt,UV_cat),axis=0)  # YUV 的存储格式为：YV12（YYYY UUVV）
       
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YV12)  # opencv不能直接读取YUV文件，需要转化注意 YUV 的存储格式
    cv2.imwrite(f'./yuv2bgr.jpg', bgr_img)
      
    fp.close()
    return img