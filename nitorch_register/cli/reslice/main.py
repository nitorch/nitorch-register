import sys
import os
import json
import torch
import nitorch_io as io
from nitorch_cli.parser import ParseError
from nitorch_core import  py
from nitorch_core.dtypes import dtype as nitype
from nitorch_core.extra import make_vector
from nitorch_fastmath import logm, expm
from nitorch_interpol import (
    pull, resize_flow, resize_affine, affine_grid, spline_coeff_nd
)
from nitorch_register.affine import (
    voxel_size as get_voxel_size,
    affine_inv, affine_matmul, affine_lmdiv, affine_matvec, affine_resize,
)
from nitorch_register.flow_inv import flow_inv
from nitorch_register.shoot import shoot
from nitorch_register.svf import exp as vexp
from .parser import parse, help
from . import struct


def entrypoint(argv=None):
    """Generic reslicing

    This is a command-line utility.
    """

    try:
        argv = argv or sys.argv[1:]
        options = parse(list(argv))
        if not options:
            return

        read_info(options)
        collapse(options)
        write_data(options)

    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


def squeeze_to_nd(dat, dim=3, channels=1):

    if isinstance(dat, (tuple, list)):
        shape = dat
        dat = None
    else:
        shape = dat.shape
    shape = list(shape)
    while len(shape) > dim + channels:
        has_deleted = False
        for d in reversed(range(dim, len(shape))):
            if shape[d] == 1:
                del shape[d]
                has_deleted = True
        if has_deleted:
            continue
        for d in reversed(range(len(shape)-channels)):
            if shape[d] == 1:
                del shape[d]
                has_deleted = True
                break
        if has_deleted:
            continue

        raise ValueError(f'Cannot squeeze shape so that it has '
                         f'{dim} spatial dimensions and {channels} '
                         f'channels.')
    shape = shape + [1] * max(0, dim-len(shape))
    if len(shape) < dim + channels:
        ones = [1] * max(0, dim+channels-len(shape))
        shape = shape[:dim] + ones + shape[dim:]
    shape = tuple(shape)
    if dat is not None:
        dat = dat.reshape(shape)
        return dat
    return shape


def read_info(options):
    """Load affine transforms and space info of other volumes"""

    def read_file(fname):
        o = struct.FileWithInfo()
        o.fname = fname
        o.dir = os.path.dirname(fname) or '.'
        o.base = os.path.basename(fname)
        o.base, o.ext = os.path.splitext(o.base)
        if o.ext in ('.gz', '.bz2'):
            zext = o.ext
            o.base, o.ext = os.path.splitext(o.base)
            o.ext += zext
        f = io.volumes.map(fname)
        o.float = nitype(f.dtype).is_floating_point
        o.shape = squeeze_to_nd(f.shape, dim=3, channels=1)
        o.channels = o.shape[-1]
        o.shape = o.shape[:3]
        o.affine = f.affine.float()
        return o

    def read_affine(fname):
        mat = io.transforms.loadf(fname).float()
        return squeeze_to_nd(mat, 0, 2)

    def read_field(fname):
        f = io.volumes.map(fname)
        return f.affine.float(), f.shape[:3]

    options.files = [read_file(file) for file in options.files]
    for trf in options.transformations:
        if isinstance(trf, struct.Linear):
            trf.affine = read_affine(trf.file)
        else:
            trf.affine, trf.shape = read_field(trf.file)
    if options.target:
        options.target = read_file(options.target)
        if options.voxel_size:
            options.voxel_size = make_vector(options.voxel_size, 3,
                                             dtype=options.target.affine.dtype)
            factor = get_voxel_size(options.target.affine) / options.voxel_size
            options.target.affine, options.target.shape = \
                affine_resize(options.target.affine, options.target.shape,
                              factor=factor, anchor='f')


def collapse(options):
    options.transformations = collapse_transforms(options.transformations)


def collapse_transforms(transformations):
    """Pre-invert affines and combine sequential affines"""
    trfs = []
    last_trf = None
    for trf in transformations:
        if isinstance(trf, struct.Linear):
            if trf.square:
                trf.affine = expm(logm(trf.affine).mul_(0.5))
            if trf.inv:
                trf.affine = affine_inv(trf.affine)
                trf.inv = False
            if isinstance(last_trf, struct.Linear):
                last_trf.affine = affine_matmul(last_trf.affine, trf.affine)
            else:
                last_trf = trf
        else:
            if isinstance(last_trf, struct.Linear):
                trfs.append(last_trf)
                last_trf = None
            trfs.append(trf)
    if isinstance(last_trf, struct.Linear):
        trfs.append(last_trf)
    return trfs


def exponentiate_transforms(transformations, **backend):
    for trf in transformations:
        if isinstance(trf, struct.Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.json:
                if trf.square:
                    trf.dat.mul_(0.5)
                with open(trf.json) as f:
                    prm = json.load(f)
                prm['voxel_size'] = get_voxel_size(trf.affine)
                trf.dat = shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv)
                if trf.inv:
                    trf.dat = trf.dat[-1]
            else:
                if trf.square:
                    trf.dat.mul_(0.5)
                trf.dat = vexp(trf.dat[None], displacement=True, inverse=trf.inv)
            trf.dat = trf.dat[0]  # drop batch dimension
            trf.inv = False
            trf.square = False
            trf.order = 1
        elif isinstance(trf, struct.Displacement):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.unit == 'mm':
                # convert mm displacement to vox displacement
                trf.dat = affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'
    return transformations



def write_data(options):

    backend = dict(dtype=torch.float32, device=options.device)

    # 1) Pre-exponentiate velocities
    for trf in options.transformations:
        if isinstance(trf, struct.Velocity):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.json:
                if trf.square:
                    trf.dat.mul_(0.5)
                with open(trf.json) as f:
                    prm = json.load(f)
                prm['voxel_size'] = get_voxel_size(trf.affine)
                trf.dat = shoot(trf.dat[None], displacement=True,
                                        return_inverse=trf.inv)
                if trf.inv:
                    trf.dat = trf.dat[-1]
            else:
                if trf.square:
                    trf.dat.mul_(0.5)
                trf.dat = vexp(trf.dat[None], displacement=True, inverse=trf.inv)
            trf.dat = trf.dat[0]  # drop batch dimension
            trf.inv = False
            trf.square = False
            trf.order = 1
        elif isinstance(trf, struct.Displacement):
            f = io.volumes.map(trf.file)
            trf.affine = f.affine
            trf.shape = squeeze_to_nd(f.shape, 3, 1)
            trf.dat = f.fdata(**backend).reshape(trf.shape)
            trf.shape = trf.shape[:3]
            if trf.unit == 'mm':
                # convert mm displacement to vox displacement
                trf.dat = affine_lmdiv(trf.affine, trf.dat[..., None])
                trf.dat = trf.dat[..., 0]
                trf.unit = 'vox'

    # 2) If the first transform is linear, compose it with the input
    #    orientation matrix
    if (options.transformations and
            isinstance(options.transformations[0], struct.Linear)):
        trf = options.transformations[0]
        for file in options.files:
            mat = file.affine.to(**backend)
            aff = trf.affine.to(**backend)
            file.affine = affine_lmdiv(aff, mat)
        options.transformations = options.transformations[1:]

    def build_from_target(affine, shape, smart=False):
        """Compose all transformations, starting from the final orientation"""
        if smart and all(isinstance(trf, struct.Linear) for trf in options.transformations):
            return None
        grid = affine_grid(affine.to(**backend), shape)
        for trf in reversed(options.transformations):
            if isinstance(trf, struct.Linear):
                grid = affine_matvec(trf.affine.to(**backend), grid)
            else:
                mat = trf.affine.to(**backend)
                if trf.inv:
                    vx0 = get_voxel_size(mat)
                    vx1 = get_voxel_size(affine.to(**backend))
                    factor = vx0 / vx1
                    oldshape = trf.dat.shape[:-1]
                    disp, mat = resize_flow(trf.dat, factor, order=trf.order)
                    mat = resize_affine(mat, oldshape, factor)
                    disp = flow_inv(disp)
                    order = 1
                else:
                    disp = trf.dat
                    order = trf.order
                imat = affine_inv(mat)
                grid = affine_matvec(imat, grid)
                grid += pull(disp, grid, order=order)
                grid = affine_matvec(mat, grid)
        return grid

    # 3) If target is provided, we can build most of the transform once
    #    and just multiply it with a input-wise affine matrix.
    if options.target:
        grid = build_from_target(options.target.affine, options.target.shape)
        oaffine = options.target.affine

    # 4) Loop across input files
    opt_pull0 = dict(order=options.interpolation,
                     bound=options.bound,
                     extrapolate=options.extrapolate)
    opt_coeff = dict(order=options.interpolation,
                     bound=options.bound,
                     dim=3,
                     inplace=True)
    output = py.make_list(options.output, len(options.files))
    for file, ofname in zip(options.files, output):
        is_label = isinstance(options.interpolation, str) and options.interpolation == 'l'
        ofname = ofname.format(dir=file.dir, base=file.base, ext=file.ext)
        print(f'Reslicing:   {file.fname}\n'
              f'          -> {ofname}')
        if is_label:
            backend_int = dict(dtype=torch.long, device=backend['device'])
            dat = io.volumes.load(file.fname, **backend_int)
            opt_pull = dict(opt_pull0)
            opt_pull['order'] = 1
        else:
            dat = io.volumes.loadf(file.fname, rand=False, **backend)
            opt_pull = opt_pull0
        dat = dat.reshape([*file.shape, file.channels])
        dat = dat.movedim(-1, 0)

        if not options.target:
            oaffine = file.affine
            oshape = file.shape
            if options.voxel_size:
                ovx = make_vector(options.voxel_size, 3, dtype=oaffine.dtype)
                factor = get_voxel_size(oaffine) / ovx
                oaffine, oshape = affine_resize(oaffine, oshape, factor=factor, anchor='f')
            grid = build_from_target(oaffine, oshape, smart=not options.voxel_size)
        if grid is not None:
            mat = file.affine.to(**backend)
            imat = affine_inv(mat)
            if options.prefilter and not is_label:
                dat = spline_coeff_nd(dat, **opt_coeff)
            dat = pull(dat, affine_matvec(imat, grid), **opt_pull)
        dat = dat.movedim(0, -1)

        if is_label:
            io.volumes.save(dat, ofname, like=file.fname, affine=oaffine, dtype=options.dtype)
        else:
            io.volumes.savef(dat, ofname, like=file.fname, affine=oaffine, dtype=options.dtype)


