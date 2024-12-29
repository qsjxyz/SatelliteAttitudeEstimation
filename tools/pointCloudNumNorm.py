import os
import tqdm
import numpy as np
from lib.utils.fps import farthest_point_sample

if __name__ == '__main__':
    inputData = '/home/d409/SatellitePoseUni6D/data/BUAA-SID'
    for object in ['a2100', 'cobe', 'early-bird', 'fengyun', 'galileo']:
        inputDir = os.path.join(inputData, object, 'pointCloud-256')
        outputDir = os.path.join(inputData, object, 'pointCloud-64')
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)

        pointNum = 0
        for filename in tqdm.tqdm(os.listdir(inputDir)):
            point_file = open(os.path.join(inputDir, filename))
            point_lines = point_file.readlines()
            point = []
            for point_line in point_lines:
                point_line = point_line.strip('\n').split(' ')
                point.append([float(point_line[0]), float(point_line[1]), float(point_line[2])])
            point = np.array(point)
            pointNum += point.shape[0]
            point = farthest_point_sample(point, 64)

            point_out_file = open(os.path.join(outputDir, filename), 'w')
            for point_one in point:
                point_out_file.writelines(str(point_one[0]) + ' ' + str(point_one[1]) + ' ' + str(point_one[2]) + '\n')
            point_out_file.close()
        print(pointNum/len(os.listdir(inputDir)))