import simplejson
import numpy as np


def norm(x):
    return np.sqrt(np.sum(x**2))


def normalized(x):
    return x/norm(x)


def placeSomaAtDepth(input_json, output_json, depth):
    with open(input_json) as fp:
        tree = simplejson.load(fp)
    treePoints = tree['treePoints']['data']
    treeLines = tree['treeLines']['data']

    # Find the soma coordinate by averaging all points on the soma contour
    somaContourTypeId = tree['customTypes']['somaContour']['id']
    somaContours = []
    for line in treeLines:
        if line[0] == somaContourTypeId:
            somaContours.append(line)
    # Often, multiple contours are (wrongly) labelled 'somaContour'
    # In all cases, the last one is the actual soma contour
    soma = somaContours[-1]
    # Compute average of all contour points
    firstPoint = soma[1]
    numPoints = soma[2]
    somaCoord = np.array([0, 0, 0], np.float64)
    for i in range(firstPoint, firstPoint+numPoints):
        pt = np.array(treePoints[i][0:3], np.float64)
        somaCoord += pt
    somaCoord /= numPoints

    # Find center of gravity of all apical dendrite line segments
    apicalDendriteTypeId = tree['customTypes']['apical']['id']
    cog = np.array([0, 0], np.double)
    cogWeight = 0.0
    for line in treeLines:
        if line[0] == apicalDendriteTypeId:
            parentLine = treeLines[line[3]]
            # the first point of the line connects to
            # the last point of the parent line
            if parentLine[0] == apicalDendriteTypeId:
                pPrev = parentLine[1] + parentLine[2] - 1 - parentLine[4]
            else:
                pPrev = None
            for p in range(line[1], line[1]+line[2]):
                if pPrev:
                    p0 = np.array(treePoints[pPrev][0:2], np.double)
                    p1 = np.array(treePoints[p][0:2], np.double)
                    diff = p1 - p0
                    normDiff = norm(diff)
                    cog += (p0/2 + p1/2) * normDiff
                    cogWeight += normDiff
                pPrev = p
    cog /= cogWeight
    # For debugging: add cog to the tree and connect to soma
    treePoints.append([cog[0], cog[1], 0, 10])
    treeLines.append([1, len(treePoints)-1, 1, 2, 0])

    # Use cog to compute rotation of the neuron
    # (assuming that the soma-to-cog vector points to pial surface)
    normalizedCog = normalized(cog-somaCoord[0:2])
    cosAngle = np.dot([normalizedCog[0], normalizedCog[1], 0], [0, 1, 0])
    if abs(normalizedCog[1]) > 0.5:
        sinAngle = cosAngle * normalizedCog[0] / normalizedCog[1]
    else:
        sinAngle = (1 - cosAngle*normalizedCog[1]) / normalizedCog[0]

    # Apply the somaCoord, rotation and depth/xshift to all points
    for pt in treePoints:
        # First subtract somaCoord, to place soma at (0,0,0)
        pt[0] = pt[0] - somaCoord[0]  # x
        pt[1] = pt[1] - somaCoord[1]  # y
        pt[2] = pt[2] - somaCoord[2]  # z
        # Next apply rotation to place cog directly above soma.
        nw0 = cosAngle*pt[0] - sinAngle*pt[1]
        nw1 = sinAngle*pt[0] + cosAngle*pt[1]
        # Finally apply depth
        pt[0] = round(nw0, 3)  # x
        pt[1] = round(nw1-depth, 3)  # y
        pt[2] = round(pt[2], 3)

    # Save the processed neuron
    with open(output_json, 'w') as fp:
        simplejson.dump(tree, fp, indent=2)


if __name__ == '__main__':
    import os.path as op
    from glob import glob

    input_folder = 'morphs_json'
    output_folder = 'morphs_atdepth_json'
    for input_json in glob(op.join(input_folder, '*.json')):
        filename = op.basename(input_json)
        print(filename)
        if not filename.startswith('2013_03'):  # 2013 is date, not depth
            output_json = op.join(output_folder, filename)
            # extract depth from file name
            depth = int(filename.split('_')[0])
            # place soma and save to output folder
            placeSomaAtDepth(input_json=input_json, output_json=output_json,
                             depth=depth)
