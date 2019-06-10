"""
Code for:
    
    Multigrid reconstruction in tomographic imaging
    D. Marlevi, H. Kohr, J-W. Buurlage, B. Gao, K. J. Batenburg, M. Colarieti-Tosti
    
Code example:

    Multiple ROIs and ROI nesting.
    
    Nested ROI example
    
Note:
    
    For proprietary reasons, the in-house foam phantom has been replaced with a 
    numerical Shepp-Logan phantom. The multigrid code is however left unchanged.
"""

# %% Import module
import numpy as np
import odl
import odl_multigrid as multigrid
import matplotlib.pyplot as plt

# %%
plt.close('all')

# %%

# Background discretization
min_pt = [-10.24, -10.24]
max_pt = [10.24, 10.24]
coarse_discr = odl.uniform_discr(min_pt, max_pt, [32, 32])

# ROI positions and discretisations
insert_min_pt = [-8, -2.5]
insert_max_pt = [3.04, 9.02]
fine_discr = odl.uniform_discr(min_pt, max_pt, [128, 128])

insert_min_pt_1 = [-7.5, 2.5]
insert_max_pt_1 = [-2.5, 7.5]
fine_discr_1 = odl.uniform_discr(min_pt, max_pt, [1024, 1024])

# Geometry
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)

# Make detector large enough to cover the object
src_radius = 50
det_radius = 50
opening_angle = np.arctan(max(np.max(np.abs(min_pt)), np.max(np.abs(max_pt))) /
                          src_radius)
det_size = np.floor(2 * (src_radius + det_radius) * np.sin(opening_angle))
det_shape = int(det_size / np.min(fine_discr.cell_sides))
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
det_partition = odl.uniform_partition(det_min_pt*1.1, det_max_pt*1.1, det_shape)
geometry = odl.tomo.FanFlatGeometry(angle_partition, det_partition,
                                    src_radius, det_radius)

# Mask
coarse_mask = multigrid.operators.MaskingOperator(coarse_discr,
                                                  insert_min_pt, insert_max_pt)

coarse_ray_trafo = odl.tomo.RayTransform(coarse_discr, geometry,
                                         impl='astra_cuda')

masked_coarse_ray_trafo = coarse_ray_trafo * coarse_mask

fine_mask_1 = multigrid.operators.MaskingOperator(fine_discr,
                                                  insert_min_pt_1, insert_max_pt_1)

fine_mask_2 = multigrid.operators.MaskingOperator(fine_discr,
                                                  insert_min_pt, insert_max_pt)

ray_trafo_fine = odl.tomo.RayTransform(fine_discr, geometry,
                                       impl='astra_cuda')

masked_ray_trafo_fine = ray_trafo_fine * (fine_mask_1-fine_mask_2)

# Insert discr
insert_discr = odl.uniform_discr_fromdiscr(fine_discr, 
                                           min_pt=insert_min_pt, 
                                           max_pt=insert_max_pt,
                                           cell_sides=fine_discr.cell_sides)

insert_discr_1 = odl.uniform_discr_fromdiscr(fine_discr_1, 
                                             min_pt=insert_min_pt_1, 
                                             max_pt=insert_max_pt_1,
                                             cell_sides=fine_discr_1.cell_sides)

# Ray trafo on the insert discretization only
insert_ray_trafo = odl.tomo.RayTransform(insert_discr, 
                                         geometry,
                                         impl='astra_cuda')

insert_mask = multigrid.operators.MaskingOperator(insert_discr,
                                                  insert_min_pt_1, insert_max_pt_1)

masked_insert_ray_trafo = insert_ray_trafo * insert_mask

insert_ray_trafo_1 = odl.tomo.RayTransform(insert_discr_1, 
                                           geometry,
                                           impl='astra_cuda')

# Forward operator = sum of masked coarse ray trafo and insert ray trafo
sum_ray_trafo = odl.ReductionOperator(masked_coarse_ray_trafo,
                                      masked_insert_ray_trafo,
                                      insert_ray_trafo_1)

fine_ray_trafo = odl.tomo.RayTransform(fine_discr_1, 
                                       geometry,
                                       impl='astra_cuda')

pspace = sum_ray_trafo.domain

# Phantom
phantom = odl.phantom.shepp_logan(fine_discr_1, modified=True)

#
resizing_operator = odl.ResizingOperator(fine_discr_1, insert_discr_1)
phantom_insert = resizing_operator(phantom)

data = fine_ray_trafo(phantom)

noisy_data = data + odl.phantom.white_noise(fine_ray_trafo.range, stddev=0.1)

# %% Reconstruction
timing = True

callback = (odl.solvers.CallbackPrintIteration(step=2) &
            odl.solvers.CallbackShow(step=2))
reco = pspace.zero()
if timing:
    callback = None
    with odl.util.Timer(reco_method):
        odl.solvers.conjugate_gradient_normal(
            sum_ray_trafo, reco, noisy_data, niter=10, callback=callback)
else:
    odl.solvers.conjugate_gradient_normal(
        sum_ray_trafo, reco, noisy_data, niter=10, callback=callback)
multigrid.graphics.show_all(reco)

# %% SSIM
from skimage.measure import compare_ssim as ssim

full_reco = fine_discr_1.zero()
odl.solvers.conjugate_gradient_normal(fine_ray_trafo,
                                      full_reco,
                                      noisy_data,
                                      niter=10,
                                      callback=callback)

full_reco_insert = resizing_operator(full_reco)
SSIM = []
Eucl = []
SSIM.append(ssim(reco[-1],full_reco_insert))
Eucl.append(np.sqrt(np.sum((full_reco_insert-reco[-1])**2)))
print("SSIM: %f, Eucl: %f\n" %(SSIM[0], Eucl[0]))