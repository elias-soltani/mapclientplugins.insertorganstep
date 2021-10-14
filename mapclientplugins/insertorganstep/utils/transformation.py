"""
A script to find the transformation matrix between input and target coordinates.
"""

import numpy as np
import csv


def align_mesh(x, y, scaling=True, reflection='best'):
    """
    align_mesh is a method that performs a Procrustes analysis which
    determines a linear transformation (translation, reflection, orthogonal rotation
    and scaling) of the nodes in mesh to best conform them to the nodes in reference_mesh,
    using the sum of squared errors as the goodness of fit criterion.
    Inputs:
    ------------
    reference_mesh, mesh
        meshes (as morphic meshes) of target and input coordinates. they must have equal
        numbers of  nodes (rows), but mesh may have fewer dimensions
        (columns) than reference_mesh.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of reference_mesh, ((reference_mesh - reference_mesh.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    self.mesh
        Aligned mesh
    :param reference_mesh
    :param mesh
    :param scaling
    :param reflection
    :return: d, Z, tform, self.mesh
    """

    import os

    print('\n\t=========================================\n')
    print('\t   ALIGNING MESH... \n')
    print('\t   PLEASE WAIT... \n')

    # r = morphic.Mesh(reference_mesh)
    # self.mesh = morphic.Mesh(mesh)

    # X = r.get_nodes()
    # Y = self.mesh.get_nodes()

    X = x
    Y = y

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    """ centred Frobenius norm """
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    """ scale to equal (unit) norm """
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    """ optimum rotation matrix of Y """
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        """ if the current solution use a reflection? """
        have_reflection = np.linalg.det(T) < 0

        """ if that's not what was specified, force another reflection """
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        """ optimum scaling of Y """
        b = traceTA * normX / normY

        """ standarised distance between X and b*Y*T + c """
        d = 1 - traceTA ** 2

        """ transformed coords """
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    """ translation matrix """
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    """ transformation values """
    tform = {'rotation': T, 'scale': b, 'translation': c}

    # for num, object in enumerate(self.mesh.nodes):
    #     node = self.mesh.nodes[object.id].values[:, 0]
    #     Zlist = Z.tolist()
    #     self.mesh.nodes[object.id].values[:, 0] = Zlist[num]

    # if self.output is not None:
    #     self.output = None

    # self.output = 'morphic_aligned'

    # mesh_output = os.path.normpath(mesh + os.sep + os.pardir)
    # mesh_output = os.path.normpath(mesh_output + os.sep + os.pardir)

    # mesh_output = os.path.join(mesh_output, self.output)
    #
    # if not os.path.exists(mesh_output):
    #     os.makedirs(mesh_output)

    # mesh_name = mesh.split('/')

    # self.mesh.save(os.path.join(mesh_output, str(mesh_name[-1])))

    # print('\t   ALIGNED MESH SAVED IN \n')
    # print('\t   %s DIRECTORY \n') % mesh_output

    print('\n\t=========================================\n')

    return d, Z, tform


def read_input_target(input_file, target_file):
    with open(input_file, 'r') as inpf, open(target_file, 'r') as tarf:
        inpr = csv.reader(inpf)
        tarr = csv.reader(tarf)
        next(inpr, None)
        next(tarr, None)
        xinput = []
        xtarget = []
        for pointx in inpr:
            xinput.append([float(c) for c in pointx[:-1]])
        for pointy in tarr:
            xtarget.append([float(c) for c in pointy[:-1]])

    xinput = np.array(xinput)
    xtarget = np.array(xtarget)

    mean_input = np.mean(xinput, axis=0)
    return xinput, xtarget, mean_input


def convert_to_transformation_matrix(tform):
    transform = np.identity(4)
    translate = np.zeros([4, 4])
    transform[:3, :3] = tform['scale']*tform['rotation']
    translate[:3, 3] = tform['translation']
    return transform, translate


def write_to_exnode(node, transform, translate, mean_input, output_file, tform=None):
    version = 2
    with open(output_file, 'w') as out:
        if version == 1:
            header = """Group name : abc
 #Fields=3
1) coordinates, coordinate, rectangular cartesian, #Components=3
  x.  Value index= 1, #Derivatives= 0
  y.  Value index= 2, #Derivatives= 0
  z.  Value index= 3, #Derivatives= 0
2) transformation_matrix, field, rectangular cartesian, #Components=16
	1.  Value index= 4, #Derivatives= 0
	2.  Value index= 5, #Derivatives= 0
	3.  Value index= 6, #Derivatives= 0
	4.  Value index= 7, #Derivatives= 0
	5. Value index=  8, #Derivatives= 0
	6. Value index=  9, #Derivatives= 0
	7. Value index= 10, #Derivatives= 0
	8. Value index= 11, #Derivatives= 0
	9. Value index= 12, #Derivatives= 0
	10. Value index= 13, #Derivatives= 0
	11. Value index= 14, #Derivatives= 0
	12. Value index= 15, #Derivatives= 0
	13. Value index= 16, #Derivatives= 0
	14. Value index= 17, #Derivatives= 0
	15. Value index= 18, #Derivatives= 0
	16. Value index= 19, #Derivatives= 0
3) translation_field, field, rectangular cartesian, #Components=16
	1. Value index=  20, #Derivatives= 0
	2. Value index=  21, #Derivatives= 0
	3. Value index=  22, #Derivatives= 0
	4. Value index=  23, #Derivatives= 0
	5. Value index=  24, #Derivatives= 0
	6. Value index=  25, #Derivatives= 0
	7. Value index=  26, #Derivatives= 0
	8. Value index=  27, #Derivatives= 0
	9. Value index=  28, #Derivatives= 0
	10. Value index= 29, #Derivatives= 0
	11. Value index= 30, #Derivatives= 0
	12. Value index= 31, #Derivatives= 0
	13. Value index= 32, #Derivatives= 0
	14. Value index= 33, #Derivatives= 0
	15. Value index= 34, #Derivatives= 0
	16. Value index= 35, #Derivatives= 0
 """
        elif version == 2:
            header = """Group name : abc
#Fields=3
1) scale, field, rectangular cartesian, #Components=1
	1.  Value index= 1, #Derivatives= 0
2) rotation, field, rectangular cartesian, #Components=9
	1.  Value index= 2, #Derivatives= 0
	2.  Value index= 3, #Derivatives= 0
	3.  Value index= 4, #Derivatives= 0
	4.  Value index= 5, #Derivatives= 0
	5. Value index=  6, #Derivatives= 0
	6. Value index=  7, #Derivatives= 0
	7. Value index= 8, #Derivatives= 0
	8. Value index= 9, #Derivatives= 0
	9. Value index= 10, #Derivatives= 0
3) translation, field, rectangular cartesian, #Components=3
	1. Value index=  11, #Derivatives= 0
	2. Value index=  12, #Derivatives= 0
	3. Value index=  13, #Derivatives= 0        
"""

        out.write(header)
        out.write('Node:     {0}\n'.format(node))
        string = '{} {} {}\n'
        if version == 1:
            out.write(string.format(mean_input[0], mean_input[1], mean_input[2]))
            for i in range(0, 15, 3):
                out.write(string.format(transform.T.reshape(-1)[i], transform.T.reshape(-1)[i+1], transform.T.reshape(-1)[i+2]))
            out.write(string.format(transform.T.reshape(-1)[-1], translate.T.reshape(-1)[0], translate.T.reshape(-1)[1]))
            for i in range(2, 14, 3):
                out.write(string.format(translate.T.reshape(-1)[i], translate.T.reshape(-1)[i+1], translate.T.reshape(-1)[i+2]))
            out.write('{} {}\n'.format(translate.T.reshape(-1)[-2], translate.T.reshape(-1)[-1]))
        elif version == 2:
            scale = tform["scale"]
            rotation = tform["rotation"].T
            translation = tform["translation"]
            out.write('{}\n'.format(scale))
            for i in range(3):
                out.write(string.format(rotation[i,0], rotation[i,1], rotation[i,2]))
            out.write(string.format(translation[0], translation[1], translation[2]))


if __name__ == "__main__":
    node = 601
    input_file = 'lung_reference.csv'
    target_file = 'trans_lung3.csv'
    output_file = 'trans_lung3.exnode'

    xinput, xtarget, mean_input = read_input_target(input_file, target_file)
    numberOfPoints = len(xinput)
    d, Z, tform = align_mesh(xtarget, xinput, scaling=True, reflection='best')
    print(d, Z)
    print(tform)
    transform, translate = convert_to_transformation_matrix(tform)
    print('transform\n', transform)
    print('translate\n', translate)
    print('s*Y*R+c')
    print(np.matmul(np.array(xinput),tform['scale']*tform['rotation'])+tform['translation'])
    print('xtarget\n',xtarget)
    print('xinputplus1 * (transform+translate.T)')
    print(np.matmul(np.concatenate((xinput, np.ones((numberOfPoints,1))), axis=1), transform+translate.T))
    write_to_exnode(node, transform, translate, mean_input, output_file, tform)
