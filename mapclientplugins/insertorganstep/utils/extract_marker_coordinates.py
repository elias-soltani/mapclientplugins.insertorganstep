import os
import csv

from opencmiss.utils.zinc.general import ChangeManager
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.finiteelement import getMaximumNodeIdentifier
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.zinc.result import RESULT_OK
from opencmiss.utils.zinc.finiteelement import findNodeWithName
from opencmiss.utils.zinc.field import findOrCreateFieldGroup, findOrCreateFieldNodeGroup, findOrCreateFieldCoordinates
from opencmiss.utils.zinc.field import findOrCreateFieldStoredMeshLocation
from transformation import read_input_target, align_mesh, convert_to_transformation_matrix, write_to_exnode
import numpy as np


class CsvToExnode:
    def __init__(self, filepath, output, nodeOffset, group_name, coordinates_name='coordinates'):
        self._fileName = os.path.basename(filepath)
        self._dirname = os.path.dirname(filepath)
        self._output = output
        self._inputFile = filepath
        self._nodeOffset = nodeOffset
        self._coordinates_name = coordinates_name
        self._group_name = group_name
        self._context = Context("CsvToExnode")
        self._region = self._context.getDefaultRegion()
        self._fieldmodule = self._region.getFieldmodule()
        self._onlyCoordinates = True

    def csv_to_exnode(self):
        self.convert_csv_to_exf()
        self.convert_exf_to_exnode()

    def convert_csv_to_exf(self):
        coordinates = findOrCreateFieldCoordinates(self._fieldmodule, components_count=3)
        cache = self._fieldmodule.createFieldcache()

        #################
        # Create nodes
        #################

        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        if not self._onlyCoordinates:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)

        nodeIdentifier = self._nodeOffset
        with open(self._inputFile, 'r') as csvf:
            reader = csv.reader(csvf)
            next(reader, None)
            for point in reader:
                if nodeIdentifier == self._nodeOffset:
                    x0 = [float(p) for p in point[0:3]]
                    nodeIdentifier += 1
                elif nodeIdentifier == self._nodeOffset + 1:
                    x1 = [float(p) for p in point[0:3]]
                    node = nodes.createNode(nodeIdentifier - 1, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x0)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x1)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    x0 = [d for d in x1]
                    nodeIdentifier = nodeIdentifier + 1
                else:
                    x1 = [float(p) for p in point[0:3]]
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x1)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    x0 = [d for d in x1]
                    nodeIdentifier = nodeIdentifier + 1

        outputFile = os.path.join(self._dirname, self._fileName.split('.')[0]+'.exf')
        self._region.writeFile(outputFile)

    def convert_exf_to_exnode(self, **kwargs):
        """
        Write model nodes to file.
        """
        filepath = kwargs['filepath'] if 'filepath' in kwargs else os.path.join(self._dirname, self._fileName.split('.')[0] + '.exf')
        filename = os.path.basename(filepath)
        context = Context('convert to exnode')
        region = context.getDefaultRegion()
        region.readFile(filepath)
        fieldmodule = region.getFieldmodule()
        coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
        coordinates.setName(self._coordinates_name)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        markerGroup = findOrCreateFieldGroup(fieldmodule, self._group_name)
        markerPoints = findOrCreateFieldNodeGroup(markerGroup, nodes).getNodesetGroup()
        is_all = fieldmodule.createFieldConstant(1)
        markerPoints.addNodesConditional(is_all)

        sir = region.createStreaminformationRegion()
        sir.setRecursionMode(sir.RECURSION_MODE_OFF)
        srm = sir.createStreamresourceMemory()
        sir.setResourceGroupName(srm, self._group_name)
        sir.setResourceDomainTypes(srm, Field.DOMAIN_TYPE_NODES)
        result = region.write(sir)
        assert result == RESULT_OK
        exnodeFileName = filename.split('.')[0] + ".exnode"
        working_directory = os.path.dirname(filepath)
        exnodeFilepath = os.path.join(working_directory, exnodeFileName)
        region.writeFile(exnodeFilepath)


class CsvToExf:
    def __init__(self, file, nodeOffset):
        fileName = os.path.basename(file)
        dirname = os.path.dirname(file)
        self._inputFile = file
        self._context = Context("CsvToEx")
        self._region = self._context.getDefaultRegion()
        self._fieldmodule = self._region.getFieldmodule()
        self._onlyCoordinates = True

        coordinates = findOrCreateFieldCoordinates(self._fieldmodule, components_count=3)
        cache = self._fieldmodule.createFieldcache()

        #################
        # Create nodes
        #################

        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        if not self._onlyCoordinates:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)

        nodeIdentifier = nodeOffset
        with open(self._inputFile, 'r') as csvf:
            reader = csv.reader(csvf)
            next(reader, None)
            for point in reader:
                if nodeIdentifier == nodeOffset:
                    x0 = [float(p) for p in point[0:3]]
                    nodeIdentifier += 1
                elif nodeIdentifier == nodeOffset + 1:
                    x1 = [float(p) for p in point[0:3]]
                    node = nodes.createNode(nodeIdentifier - 1, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x0)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x1)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    x0 = [d for d in x1]
                    nodeIdentifier = nodeIdentifier + 1
                else:
                    x1 = [float(p) for p in point[0:3]]
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x1)
                    if not self._onlyCoordinates:
                        dx_ds1 = [x1[c] - x0[c] for c in range(3)]
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                    x0 = [d for d in x1]
                    nodeIdentifier = nodeIdentifier + 1

        outputFile = os.path.join(dirname, fileName.split('.')[0]+'.exf')
        self._region.writeFile(outputFile)


class EmbedOrgan:
    """
    A class for extracting organ marker coordinates and embed the fiducial markers in the body-coordinates
    """

    def __init__(self, organ_name, organ_filepath, working_directory, *body_filepath):
        self._organ_name = organ_name
        self._working_directory = working_directory
        self._organ_filepath = organ_filepath
        self._body_filepath = body_filepath
        self._body_exnodefilepath = body_filepath[0]
        self._organ_fiducial_markers = None

        # self.get_organ_fiducials(organ_filename)
        # context = Context('extract_marker_coordinates')
        # self._region = context.getDefaultRegion()
        # self._organ_dirname = organ_dirname
        # organ_file = organ_filename
        # self._region.readFile(organ_file)
        # self._fieldmodule = self._region.getFieldmodule()
        # self._mesh = self.get_highest_dimension_mesh()
        # self._nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # self._coordinates = self._fieldmodule.findFieldByName('coordinates').castFiniteElement()
        # self._markerNodes = None

        # organ_reference.exf
        # markerGroup = self._fieldmodule.findFieldByName("marker")
        # if markerGroup.isValid():
        #     markerGroup = markerGroup.castGroup()
        #     markerNodeGroup = markerGroup.getFieldNodeGroup(self._nodes)
        #     if markerNodeGroup.isValid():
        #         self._markerNodes = markerNodeGroup.getNodesetGroup()
        #
        # if self._markerNodes and (self._markerNodes.getSize() > 0):
        #     filename = os.path.basename(organ_filename)
        #     markerFilename = filename.split('.')[0] + "_marker.csv"
        #     markerFilePath = os.path.join(organ_dirname, markerFilename)
        #     # markerFilename = os.path.splitext(organ_file)[0] + "_marker.csv"
        #     with open(markerFilePath, 'w') as outstream:
        #         self._writeMarkers(outstream)

        # self.read_landmarks()
        # self.get_marker_location()
        # markerGroup2 = self._fieldmodule.findFieldByName("marker").castGroup()
        # markerNodes = markerGroup2.getFieldNodeGroup(self._nodes).getNodesetGroup()
        # markerName = self._fieldmodule.findFieldByName("marker_name")
        # assert markerName.isValid()
        # markerLocation = self._fieldmodule.findFieldByName("marker_location")
        # markerLocation2 = self._fieldmodule.createFieldFindMeshLocation(self._markerCoordinates, self._coordinates, self._mesh)
        # a = markerLocation2.setManaged(True)
        # markerLocation2.setName("Elias")
        # # assert self._markerLocation.isValid(), 'invlaide marker location'
        # # test apex marker point
        # cache = self._fieldmodule.creorgan_filepathateFieldcache()
        # node = findNodeWithName(markerNodes, markerName, "apex of heart")
        # assert node.isValid(), 'node is invalid'
        # cache.setNode(node)
        # nodeId = node.getIdentifier()
        # element, xi = markerLocation2.evaluateMeshLocation(cache, 3)
        # elementId = element.getIdentifier()
        # self._region.writeFile('rat_body_data500.exf')
        a=1

    def get_maximum_node_identifier(self):
        context = Context('node number')
        region = context.getDefaultRegion()
        region.readFile(self._body_exnodefilepath)
        fieldmodule = region.getFieldmodule()
        coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._maximum_nodeId = getMaximumNodeIdentifier(nodes)
        return self._maximum_nodeId

    def set_organ_filepath(self, organ_filepath):
        self._organ_filepath = organ_filepath

    def get_marker_coordinates(self, outputpath, *filepath, **kwargs):
        working_directory = kwargs['working_directory'] if 'working_directory' in kwargs else self._working_directory
        if 'marker_list' in kwargs:
            marker_list = kwargs['marker_list']
        else:
            marker_list = []

        context = Context('organ_fiducial_coordinates')
        region = context.getDefaultRegion()
        for f in filepath:
            region.readFile(f)
        fieldmodule = region.getFieldmodule()
        coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        coordinatesCount = coordinates.getNumberOfComponents()
        cache = fieldmodule.createFieldcache()

        markerLocation = fieldmodule.findFieldByName("marker_location")
        markerName = fieldmodule.findFieldByName("marker_name")
        markerGroup = fieldmodule.findFieldByName("marker")

        markers = []
        if markerGroup.isValid():
            markerGroup = markerGroup.castGroup()
            markerNodeGroup = markerGroup.getFieldNodeGroup(nodes)
            if markerNodeGroup.isValid():
                markerNodes = markerNodeGroup.getNodesetGroup()

        if markerLocation.isValid() and markerName.isValid():
            with ChangeManager(fieldmodule):
                markerCoordinates = fieldmodule.createFieldEmbedded(coordinates, markerLocation)
                nodeIter = markerNodes.createNodeiterator()
                node = nodeIter.next()
                with open(outputpath, 'w') as outstream:
                    outstream.write('x,y,z,landmark\n')
                    while node.isValid():
                        cache.setNode(node)
                        result, x = markerCoordinates.evaluateReal(cache, coordinatesCount)
                        if result == RESULT_OK:
                            name = markerName.evaluateString(cache)
                            if (name in marker_list) or (marker_list == []):
                                markers.append({'landmark': name, 'x': x})
                                outstream.write(",".join(str(s) for s in x) + "," + name + "\n")
                        node = nodeIter.next()

        return markers

    def get_organ_fiducials(self, organ_filename):
        context = Context('extract_marker_coordinates')

    def read_landmarks(self, file):
        # file = r'C:\Users\egha355\Desktop\embed\modify_heart\whole-heart2.exf'
        # with open(file, 'r') as lmf:
        #     reader = csv.reader(lmf)
        #     next(reader, None)
        #     landmark_name = []
        #     landmark_coordinates = []
        #     for point in reader:
        #         landmark_name.append(point[0])
        #         landmark_coordinates.append([float(point[i]) for i in range(1, 4)])
        #     self._landmark_name = landmark_name
        #     self._landmark_coordinates = landmark_coordinates

        with open(file, 'r') as fsc:
            reader = csv.reader(fsc)
            next(reader, None)
            markers = []
            for landmark in reader:
                markers.append({'landmark': landmark[-1], 'x':landmark[:-1]})

        self._organ_landmarks = markers


    def get_highest_dimension_mesh(self):
        for dimension in range(3, 1, -1):
            mesh = self._fieldmodule.findMeshByDimension(dimension)
            if mesh.getSize() > 0:
                break
        return mesh

    # def _writeMarkers(self, outstream):
    #     coordinatesCount = self._coordinates.getNumberOfComponents()
    #     cache = self._fieldmodule.createFieldcache()
    #     # cache2 = self._fieldmodule.createFieldcache()
    #     markerLocation = self._fieldmodule.findFieldByName("marker_location")
    #     markerName = self._fieldmodule.findFieldByName("marker_name")
    #
    #     # organ_marker_coordinates = self._fieldmodule.createFieldFiniteElement(3)
    #     # nodetemplate = self._markerNodes.createNodetemplate()
    #     # nodetemplate.defineField(organ_marker_coordinates)
    #     # nodetemplate.setValueNumberOfVersions(organ_marker_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    #     # nodeIdentifier = 8000100
    #
    #     if markerLocation.isValid() and markerName.isValid():
    #         with ChangeManager(self._fieldmodule):
    #             markerCoordinates = self._fieldmodule.createFieldEmbedded(self._coordinates, markerLocation)
    #             nodeIter = self._markerNodes.createNodeiterator()
    #             node = nodeIter.next()
    #             while node.isValid():
    #                 cache.setNode(node)
    #                 result, x = markerCoordinates.evaluateReal(cache, coordinatesCount)
    #
    #                 # node2 = self._markerNodes.createNode(nodeIdentifier, nodetemplate)
    #                 # cache2.setNode(node2)
    #                 # organ_marker_coordinates.setNodeParameters(cache2, -1, Node.VALUE_LABEL_VALUE, 1, x)
    #                 if result == RESULT_OK:
    #                     name = markerName.evaluateString(cache)
    #                     outstream.write(",".join(str(s) for s in x) + "," + name + "\n")
    #                 node = nodeIter.next()
    #
    #                 # nodeIdentifier = nodeIdentifier + 1
    #             self._markerCoordinates = markerCoordinates



    # def _write_marker_locations(self, outstream):
    #     coordinatesCount = self._coordinates.getNumberOfComponents()
    #     cache = self._fieldmodule.createFieldcache()
    #     markerLocation = self._fieldmodule.findFieldByName("marker_location")
    #     markerName = self._fieldmodule.findFieldByName("marker_name")
    #     if markerLocation.isValid() and markerName.isValid():
    #         with ChangeManager(self._fieldmodule):
    #             markerCoordinates = self._fieldmodule.createFieldEmbedded(self._coordinates, markerLocation)
    #             nodeIter = self._markerNodes.createNodeiterator()
    #             node = nodeIter.next()
    #             while node.isValid():
    #                 cache.setNode(node)
    #                 result, x = markerCoordinates.evaluateReal(cache, coordinatesCount)
    #                 if result == RESULT_OK:
    #                     name = markerName.evaluateString(cache)
    #                     outstream.write(",".join(str(s) for s in x) + "," + name + "\n")
    #                 node = nodeIter.next()
    #             self._markerCoordinates = markerCoordinates.castFiniteElement()

    # def get_marker_location(self):
    #     self._markerLocation = self._fieldmodule.createFieldFindMeshLocation(self._markerCoordinates, self._coordinates, self._mesh)
    #     assert RESULT_OK == self._markerLocation.setSearchMesh(self._mesh)
    #     self._markerLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_EXACT)

    def get_element_xi(self, bodyNodeFile, bodyElemFile, organ_finalstate):
        context = Context('convert to exnode')
        region = context.getDefaultRegion()

        # for f in body_filepath:
        #     region.readFile(f)
        region.readFile(bodyNodeFile)
        region.readFile(bodyElemFile)
        region.readFile(organ_finalstate)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # TODO make it possible to get the bodyFile instead of this. Note becasue organ file has markers the body file should not have any marker.
        organ_marker_coordinates = fieldmodule.findFieldByName(organ_name + '_marker_coordinates').castFiniteElement()
        organ_locations = fieldmodule.createFieldFindMeshLocation(organ_marker_coordinates, coordinates, mesh3d)
        organ_locations.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_EXACT)

        # markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        # markerNodes = markerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        # markerName = fieldmodule.findFieldByName("marker_name")

        organMarkerGroup = fieldmodule.findFieldByName(organ_name + '_marker').castGroup()
        organMarkerNodesetGroup = organMarkerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        # markerName = fieldmodule.findFieldByName("marker_name")

        cache = fieldmodule.createFieldcache()
        nodeIter = organMarkerNodesetGroup.createNodeiterator()
        node = nodeIter.next()

        element_xi_filepath = os.path.join(self._working_directory, os.path.basename(organ_finalstate).split('_')[0] + '_elementxi.csv')

        with open(element_xi_filepath, 'w', newline='') as elxif:
            writer = csv.writer(elxif)
            writer.writerow(['marker', 'element', 'xi1', 'xi2', 'xi3'])
            co = 0
            while node.isValid():
                cache.setNode(node)
                element, xi = organ_locations.evaluateMeshLocation(cache, 3)
                print('Node: ', str(co+403)+'\n', element.getIdentifier(), xi[0], xi[1], xi[2], '\n', '"'+self._organ_landmarks[co]['landmark']+'"')
                writer.writerow([self._organ_landmarks[co]['landmark'], element.getIdentifier(), xi[0], xi[1], xi[2]])
                co += 1
                node = nodeIter.next()

        # How to add the new marker locations?
        # markerGroup = fieldmodule.findFieldByName("marker_location").castGroup()
        # markerNodes = markerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        # markerName = fieldmodule.findFieldByName("marker_name")

        # get marker names

    # def writeReference(self, organ_filename, organ_name):
    #     context = Context('organ_reference')
    #     region = context.getDefaultRegion()
    #     fieldmodule = region.getFieldmodule()
    #     region.readFile(organ_filename)
    #     coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
    #     coordinatesCount = coordinates.getNumberOfComponents()
    #     cache = fieldmodule.createFieldcache()
    #     markerLocation = fieldmodule.findFieldByName("marker_location")
    #     markerName = fieldmodule.findFieldByName("marker_name")
    #
    #     nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    #
    #     # organ_reference.exf
    #     markerGroup = fieldmodule.findFieldByName("marker")
    #     if markerGroup.isValid():
    #         markerGroup = markerGroup.castGroup()
    #         markerNodeGroup = markerGroup.getFieldNodeGroup(nodes)
    #         if markerNodeGroup.isValid():
    #             markerNodes = markerNodeGroup.getNodesetGroup()
    #     # organ_name = 'heart'
    #     working_directory = os.path.join(os.path.dirname(organ_filename), organ_name)
    #     filepath = os.path.join(working_directory, organ_name + '_reference.csv')
    #
    #     with open(filepath, 'w', newline='') as reff:
    #         writer = csv.writer(reff)
    #         writer.writerow(['x', 'y', 'z', 'landmark'])
    #         if markerLocation.isValid() and markerName.isValid():
    #             with ChangeManager(fieldmodule):
    #                 markerCoordinates = fieldmodule.createFieldEmbedded(coordinates, markerLocation)
    #                 nodeIter = markerNodes.createNodeiterator()
    #                 node = nodeIter.next()
    #                 while node.isValid():
    #                     cache.setNode(node)
    #                     result, x = markerCoordinates.evaluateReal(cache, coordinatesCount)
    #
    #                     if result == RESULT_OK:
    #                         name = markerName.evaluateString(cache)
    #                         x.append(name)
    #                         writer.writerow(x)
    #                         # outstream.write(",".join(str(s) for s in x) + "," + name + "\n")
    #                     node = nodeIter.next()


    # def findCoordInOtherTimes():
    #     # First read the files with different times.
    #     context = Context('Find_coordinates')
    #     region = context.getDefaultRegion()
    #     timeKeeper = context.getTimekeepermodule()
    #     sir = region.createStreaminformationRegion()
    #     sir.setAttributeReal(sir.ATTRIBUTE_TIME, 0.0)
    #     result = region.read(sir)
    #     assert result == RESULT_OK
    #     parentDirectory = r'C:\Users\egha355\Desktop\embed\modify_heart'
    #     filename = 'bodycoordAligned_node4.exf'
    #     filepath = os.path.join(parentDirectory, filename)
    #     region.readFile(filepath)
    #     sir.setAttributeReal(sir.ATTRIBUTE_TIME, 1.0)
    #     a=1
    #
    # # Find heart marker coordinates in different time steps.
    # # Fieldmodule::createFieldTimeLookup	(	const Field & 	sourceField,
    # # const Field & 	timeField
    # # )
    # # :Fieldmodule::createFieldConstant	(	int 	valuesCount,
    # # const double * 	valuesIn
    # # )
    # # gfx def field marker_coordinates embedded field coordinates element_xi marker_location
    # # gfx define field heart/reftime constant 3.0;
    # # gfx define field heart/coordinates_reftime coordinate_system rectangular_cartesian time_lookup field coordinates time_field reftime;

    # def findCoordInOtherTimes2(self, exnode, exelem, organ_name, parentDirectory, working_directory, target_filename, final_statecsv):
    #     context = Context('Find_coordinates')
    #     region = context.getDefaultRegion()
    #     fieldmodule = region.getFieldmodule()
    #     # parentDirectory = r'C:\Users\egha355\Desktop\embed\modify_heart'
    #     markerFilename = 'rat1_fit1_node_static.exf'
    #     markerFilepath = os.path.join(parentDirectory, markerFilename)
    #
    #     result0 = region.readFile(exnode)
    #     result1 = region.readFile(exelem)
    #     result2 = region.readFile(markerFilepath)
    #
    #     coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
    #     coordinatesCount = coordinates.getNumberOfComponents()
    #     cache = fieldmodule.createFieldcache()
    #     markerLocation = fieldmodule.findFieldByName("marker_location")
    #     markerName = fieldmodule.findFieldByName("marker_name")
    #
    #     nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    #
    #     with open(final_statecsv, 'r') as fsc:
    #         reader = csv.reader(fsc)
    #         next(reader, None)
    #         marker_name = []
    #         for landmark in reader:
    #             marker_name.append(landmark[-1])
    #
    #     # organ_reference.exf
    #     markerGroup = fieldmodule.findFieldByName("marker")
    #     if markerGroup.isValid():
    #         markerGroup = markerGroup.castGroup()
    #         markerNodeGroup = markerGroup.getFieldNodeGroup(nodes)
    #         if markerNodeGroup.isValid():
    #             markerNodes = markerNodeGroup.getNodesetGroup()
    #     # working_directory = os.path.join(parentDirectory, organ_name)
    #     # filepath = os.path.join(working_directory, 'trans_{}.csv'.format(organ_name))
    #     filepath = target_filename
    #
    #     with open(filepath, 'w', newline='') as trf:
    #         writer = csv.writer(trf)
    #         writer.writerow(['x', 'y', 'z', 'landmark'])
    #         if markerLocation.isValid() and markerName.isValid():
    #             with ChangeManager(fieldmodule):
    #                 markerCoordinates = fieldmodule.createFieldEmbedded(coordinates, markerLocation)
    #                 nodeIter = markerNodes.createNodeiterator()
    #                 node = nodeIter.next()
    #                 while node.isValid():
    #                     cache.setNode(node)
    #                     result, x = markerCoordinates.evaluateReal(cache, coordinatesCount)
    #
    #                     if result == RESULT_OK:
    #                         name = markerName.evaluateString(cache)
    #                         if name in marker_name:
    #                             x.append(name)
    #                             writer.writerow(x)
    #                         # outstream.write(",".join(str(s) for s in x) + "," + name + "\n")
    #                     node = nodeIter.next()

    def compute_transformation_matrix(self, working_directory, reference_filename, target_filename):
        # working_directory = r'C:\Users\egha355\Desktop\embed\modify_heart\heart'
        # reference_filename = 'heart_reference.csv'
        # target_filename = 'trans_heart0.csv'
        reference_filepath = os.path.join(working_directory, reference_filename)
        target_filepath = os.path.join(working_directory, target_filename)
        node = 601

        output_file =os.path.join(working_directory, os.path.basename(target_filename).split('.')[0] + '.exnode')

        xinput, xtarget, mean_input = read_input_target(reference_filepath, target_filepath)
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

    def embed_organ(self, organ_finalstatepath, organ_filepath, bodyNodeFile, bodyElemFile):
        nodeOffset = self.get_maximum_node_identifier() + 100
        organ_finalstateexnode = organ_finalstatepath.split('.')[0] + '.exnode'
        csvToExnode = CsvToExnode(organ_finalstatepath, organ_finalstateexnode, nodeOffset,
                                  self._organ_name + '_marker',
                                  coordinates_name=self._organ_name + '_marker_coordinates')
        csvToExnode.csv_to_exnode()
        self.read_landmarks(organ_finalstatepath)
        # self.get_element_xi(bodyNodeFile, bodyElemFile, organ_finalstateexnode)
        self.get_element_xi(bodyNodeFile, bodyElemFile, organ_finalstateexnode)

    def transform_organ(self, outputpath, *body_filepath):
        marker_list = self.get_marker_list()
        self.get_marker_coordinates(organ_referencepath, organ_filepath, marker_list=marker_list)
        # exelem = os.path.join(parentDirectory, 'rat1_fit1_element.exf')
        # exnode = os.path.join(parentDirectory, 'bodycoordAligned_node4.exf')
        # target_filename = os.path.join(working_directory, 'trans_{}0.csv'.format(organ_name))
        embedorgan.get_marker_coordinates(outputpath, *body_filepath, marker_list=marker_list)
        # findCoordInOtherTimes2(exnode, exelem, organ_name, parentDirectory, working_directory, target_filename, finalstatePath)
        target_filename = outputpath
        embedorgan.compute_transformation_matrix(working_directory, organ_name + '_reference.csv', target_filename)

    def get_marker_list(self):
        marker_list = []
        for landmark in self._organ_landmarks:
            marker_list.append(landmark['landmark'])

        return marker_list

# organ_name = 'brainstem'
# organ_filepath = r'C:\Users\egha355\Desktop\embed\human_brainstem\HumanBrainstemFit.exf'
# working_directory = r'C:\Users\egha355\Desktop\embed\human_brainstem\brainstem'
# parentDirectory = r'C:\Users\egha355\Desktop\embed\human_brainstem'
# bodyNodeFilepath = os.path.join(parentDirectory, 'human1_fit1_node1.exf')
# bodyElemFilepath = os.path.join(parentDirectory, 'human1_fit1_element.exf')
# #
# # body_nodefilepath = os.path.join(parentDirectory, 'rat1_fit1_node1.exf')
# finalstatefilename = organ_name + '_finalstate.csv'
# reference_filename = organ_name + '_reference.csv'
# # organ_finalstate = os.path.join(working_directory, organ_name + '_finalstate.exf')
# # organ_finalstateexnode = os.path.join(working_directory, organ_name + '_finalstate.exnode')
# organ_finalstatepath = os.path.join(working_directory, finalstatefilename)
# organ_referencepath = os.path.join(working_directory, reference_filename)
# dirname = os.path.dirname(organ_filepath)
# marker_filename = 'human1_fit1_node_static.exf'
# marker_filepath = os.path.join(parentDirectory, marker_filename)
#
# try:
#     organ_dirname = os.path.join(dirname, organ_name)
#     os.mkdir(organ_dirname)
# except OSError as error:
#     print(error)


organ_name = 'heart'
organ_filepath = r'C:\Users\egha355\Desktop\embed\embedding_static\mouse-with-organs\others\heart_transformed.exf'
working_directory = r'C:\Users\egha355\Desktop\embed\embedding_static\mouse-with-organs\others\heart'
parentDirectory = r'C:\Users\egha355\Desktop\embed\embedding_static\mouse-with-organs\others'
bodyNodeFilepath = os.path.join(parentDirectory, 'whole-body_node.exf')
bodyElemFilepath = os.path.join(parentDirectory, 'whole-body_element.exf')
#
# body_nodefilepath = os.path.join(parentDirectory, 'rat1_fit1_node1.exf')
finalstatefilename = organ_name + '_finalstate.csv'
reference_filename = organ_name + '_reference.csv'
# organ_finalstate = os.path.join(working_directory, organ_name + '_finalstate.exf')
# organ_finalstateexnode = os.path.join(working_directory, organ_name + '_finalstate.exnode')
organ_finalstatepath = os.path.join(working_directory, finalstatefilename)
organ_referencepath = os.path.join(working_directory, reference_filename)
dirname = os.path.dirname(organ_filepath)
marker_filename = 'human1_fit1_node_static.exf'
marker_filepath = os.path.join(parentDirectory, marker_filename)

try:
    organ_dirname = os.path.join(dirname, organ_name)
    os.mkdir(organ_dirname)
except OSError as error:
    print(error)

embedorgan = EmbedOrgan(organ_name, organ_filepath, organ_dirname, bodyNodeFilepath)
# Get the organ_finalstate.csv from fitted organ. This is only for fitted ones. For other cases the file should be given.
# embedorgan.get_marker_coordinates(organ_finalstatepath, organ_filepath)

embedorgan.embed_organ(organ_finalstatepath, organ_filepath, bodyNodeFilepath, bodyElemFilepath)
# embedorgan.get_marker_coordinates(finalstatePath, organ_filename)
# nodeOffset = embedorgan.get_maximum_node_identifier() + 100
# csvToExnode = CsvToExnode(finalstatePath, organ_finalstateexnode, nodeOffset, organ_name + '_marker', coordinates_name=organ_name+'_marker_coordinates')
# csvToExnode.csv_to_exnode()


# embedorgan.read_landmarks(organ_finalstatepath)
# embedorgan.get_element_xi(bodyNodeFile, bodyElemFile, organ_finalstateexnode)
# Copy element_xi to static file where we have marker and marker_location
# embedorgan.get_marker_coordinates(organ_referencepath, organ_filepath)
#
# marker_list = []
# for landmark in embedorgan._organ_landmarks:
#     marker_list.append(landmark['landmark'])

exelem = os.path.join(parentDirectory, 'human1_fit1_element.exf')
exnode = os.path.join(parentDirectory, 'align_node2.exf')
outputpath = os.path.join(working_directory, 'trans_{}0.csv'.format(organ_name))
embedorgan.transform_organ(outputpath, exnode, exelem, marker_filepath)
# embedorgan.get_marker_coordinates(target_filename, exnode, exelem, markerFilepath, marker_list=marker_list)
# findCoordInOtherTimes2(exnode, exelem, organ_name, parentDirectory, working_directory, target_filename, finalstatePath)
# embedorgan.compute_transformation_matrix(working_directory, organ_name + '_reference.csv', target_filename)

exnode = os.path.join(parentDirectory, 'scaffold_node3.exf')
outputpath = os.path.join(working_directory, 'trans_{}1.csv'.format(organ_name))
embedorgan.transform_organ(outputpath, exnode, exelem, marker_filepath)
# embedorgan.get_marker_coordinates(target_filename, exnode, exelem, markerFilepath, marker_list=marker_list)
# embedorgan.compute_transformation_matrix(working_directory, organ_name + '_reference.csv', target_filename)

exnode = os.path.join(parentDirectory, 'human1_fit1_node1.exf')
outputpath = os.path.join(working_directory, 'trans_{}2.csv'.format(organ_name))
embedorgan.transform_organ(outputpath, exnode, exelem, marker_filepath)
# embedorgan.get_marker_coordinates(target_filename, exnode, exelem, marker_filepath, marker_list=marker_list)
# embedorgan.compute_transformation_matrix(working_directory, organ_name + '_reference.csv', target_filename)

# exnode = os.path.join(parentDirectory, 'align_node2.exf')
# outputpath = os.path.join(working_directory, 'trans_{}3.csv'.format(organ_name))
# embedorgan.transform_organ(outputpath, exnode, exelem, marker_filepath)
# embedorgan.get_marker_coordinates(target_filename, exnode, exelem, marker_filepath, marker_list=marker_list)
# embedorgan.compute_transformation_matrix(working_directory, organ_name + '_reference.csv', target_filename)
