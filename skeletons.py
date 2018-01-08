import numpy as np
import matplotlib.pyplot as plt
# import vtk

import pickle

from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_hit_or_miss

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import center_of_mass

from scipy.spatial import distance, ConvexHull, Voronoi
from scipy.special import sph_harm


def thinningElements():
    elem = np.zeros((3, 3), dtype='bool')
    elem[2,:] = 1
    elem[1,1] = 1
    
    elemConv = np.zeros((3, 3), dtype='bool')
    elemConv[0,:] = 1
    
    elem2 = np.zeros((3, 3), dtype='bool')
    elem2[0,1] = 1
    elem2[1,1] = 1
    elem2[1,2] = 1

    elem2Conv = np.zeros((3, 3), dtype='bool')
    elem2Conv[1,0] = 1
    elem2Conv[2,0] = 1
    elem2Conv[2,1] = 1
    
    rotatedElements = np.zeros((4,4,3,3), dtype='bool')

    for r in range(4):
        rotatedElements[r,0,:,:] = np.rot90(elem, r)
        rotatedElements[r,1,:,:] = np.rot90(elemConv, r)
        rotatedElements[r,2,:,:] = np.rot90(elem2, r)
        rotatedElements[r,3,:,:] = np.rot90(elem2Conv, r)
        
    return rotatedElements


def hitOrMissThinning(lesion, thinningElements):
    img = np.zeros((60, 256, 256), dtype='bool')
    
    for point in lesion:
        img[point[0], point[1], point[2]] = 1

    for z in range(img.shape[0]):
        iterations = 0
        numSkelePoints = 0
        
        while not numSkelePoints == np.sum(img[z,:,:]):
            numSkelePoints = np.sum(img[z,:,:])
            for r in range(4):
                remove = binary_hit_or_miss(img[z,:,:], thinningElements[r,0,:,:], thinningElements[r,1,:,:])
                img[z,:,:] = img[z,:,:] - remove
                
            for r in range(4):
                remove = binary_hit_or_miss(img[z,:,:], thinningElements[r,2,:,:], thinningElements[r,3,:,:])
                img[z,:,:] = img[z,:,:] - remove
            
            iterations += 1

    print(np.sum(img), '/', len(lesion), 'lesion voxels')
    
    skeletonPoints = np.transpose(np.nonzero(img))
    return img, skeletonPoints


def voroSkeleton(lesion):
    skeleton = []

    vor = Voronoi(lesion)
        
    for region in vor.regions:
        if region.all() >= 0:
            for pointIndex in region:
                skeleton.append(vor.vertices[pointIndex])
                
    return skeleton


def getLesionSkeleton(scan):

    thinningOperators = thinningElements()

    for lesion in scan.lesionList:
        if len(lesion) > 50:
            skeleImg, hitMissSkele = hitOrMissThinning(lesion, thinningOperators)
                    
            img = np.zeros((256, 256, 60), dtype='float')
                    
            for point in lesion:
                img[point[0], point[1], point[2]] = 1
                
            
            boundaryDistance = distance_transform_edt(img, sampling=[1, 1, 3])
            
            point = center_of_mass(img)
            
            centrePoint = (int(point[0]), int(point[1]), int(point[2]))
            distanceGrad = np.abs(np.gradient(boundaryDistance))
            
            sumGrads = distanceGrad[0] + distanceGrad[1] + distanceGrad[2]
            sumGrads = np.multiply(img, sumGrads)
                
            displaySkeleton3D(lesion, hitMissSkele)
            
            plt.subplot(1, 3, 1)     
            plt.axis('off')
            plt.imshow(img[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')            
            plt.xlabel('lesion slice')
            
            plt.subplot(1, 3, 2)     
            plt.axis('off')
            plt.imshow(boundaryDistance[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')
            plt.xlabel('boundary distance')

            plt.subplot(1, 3, 3)     
            plt.axis('off')
            plt.imshow(skeleImg[centrePoint[0], centrePoint[1]-10:centrePoint[1]+10, centrePoint[2]-10:centrePoint[2]+10], cmap = plt.cm.gray, interpolation = 'nearest')            
            plt.xlabel('skeleton')
            
            plt.show()


def displaySkeleton3D(lesion, skeleton):
    
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    points2 = vtk.vtkPoints()
    vertices2 = vtk.vtkCellArray()     

    
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    Colors2 = vtk.vtkUnsignedCharArray()
    Colors2.SetNumberOfComponents(3)
    Colors2.SetName("Colors2")

    for point in lesion:
        pointId = points.InsertNextPoint(point)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pointId)
        Colors.InsertNextTuple3(255,255,255)

    for point in skeleton:
        pointId = points2.InsertNextPoint(point)
        vertices2.InsertNextCell(1)
        vertices2.InsertCellPoint(pointId)
        Colors2.InsertNextTuple3(0,255,0)

    poly = vtk.vtkPolyData()
    poly2 = vtk.vtkPolyData()

    poly.SetPoints(points)
    poly.SetVerts(vertices)
    poly.GetPointData().SetScalars(Colors)
    poly.Modified()
    poly.Update()

#    delaunay = vtk.vtkDelaunay2D()
#    delaunay.SetInput(poly)
#    delaunay.SetSource(poly)
#    delaunay.SetAlpha(0.5)
#    delaunay.Update()
#    
#    delMapper = vtk.vtkDataSetMapper()
#    delMapper.SetInputConnection(delaunay.GetOutputPort())
#    
#    delActor = vtk.vtkActor()
#    delActor.SetMapper(delMapper)
#    delActor.GetProperty().SetInterpolationToFlat()
#    delActor.GetProperty().SetRepresentationToWireframe()

    poly2.SetPoints(points2)
    poly2.SetVerts(vertices2)
    poly2.GetPointData().SetScalars(Colors2)
    poly2.Modified()
    poly2.Update()
    
#    poly3.SetPoints(points3)
#    poly3.SetVerts(vertices3)
#    poly3.GetPointData().SetScalars(Colors3)
#    poly3.Modified()
#    poly3.Update()
    
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
     
    renWin.SetSize(500, 500)

    mapper = vtk.vtkPolyDataMapper()
    mapper2 = vtk.vtkPolyDataMapper()
#    mapper3 = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly)
    mapper2.SetInput(poly2)
#    mapper3.SetInput(poly3)
    
    
    transform1 = vtk.vtkTransform()
    transform1.Translate(0.0, 0.1, 0.0)
    transform2 = vtk.vtkTransform()
    transform2.Translate(0.0, 0.0, 0.1)    
    
#    transform = vtk.vtkTransform()
#    transform.Translate(0.2, 0.0, 0.0)
#    axesTransform = vtk.vtkTransform()
#    axesTransform.Scale(0.1, 0,0)     

#    axes = vtk.vtkAxesActor()
#    axes.SetUserTransform(transform)
#    axes.SetUserTransform(axesTransform)
#    axes.AxisLabelsOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.SetUserTransform(transform1)
    actor2.GetProperty().SetPointSize(5)

#    actor3 = vtk.vtkActor()
#    actor3.SetMapper(mapper3)
#    actor3.SetUserTransform(transform2)
#    actor3.GetProperty().SetPointSize(5)
    
    ren.AddActor(actor)
    ren.AddActor(actor2)
#    ren.AddActor(axes)
#    ren.AddActor(actor3)
#    ren.AddActor(delActor)
    ren.SetBackground(.2, .3, .4)
    
    renWin.Render()
    iren.Start()

           
def displaySkeletons():
    infile = open('/usr/local/data/adoyle/mri_list.pkl', 'rb')
    mri_list = pickle.load(infile)
    infile.close()

    for scan in mri_list[0:100]:
        getLesionSkeleton(scan)


def main():
    displaySkeletons()
    
    
if __name__ == "__main__":
    main()