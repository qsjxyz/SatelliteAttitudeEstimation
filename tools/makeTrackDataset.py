import random
import os
import shutil

def sgn(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

if __name__ == '__main__':
    dataDir = 'F:/Data/BUAA-SID/data/'
    outDir = 'F:/Data/BUAA-SID/data/track/'
    clses = ['a2100', 'cobe', 'early-bird', 'fengyun', 'galileo']
    imgNumber = 200
    trackNum = 5

    for cls in clses:
        imgDir = os.path.join(dataDir, 'BUAA-SID', cls, 'image')
        labelDir = os.path.join(dataDir, 'label', '1-1-18', '01', cls, 'test.txt')
        lines = open(labelDir).readlines()
        labels2Name = {}
        for line in lines:
            line = line.strip(' \n')
            name = line.split(' ')[0]
            label = line.split(' ')[1:]
            label = " ".join(label)
            labels2Name[label] = name
        for track in range(trackNum):
            outDirFinal = os.path.join(outDir, cls, str(track))
            if not os.path.isdir(outDirFinal):
                os.makedirs(outDirFinal)
            startXYZ = [int(random.randint(-58, 58)) * 3,
                        int(random.randint(-10, 10)) * 3,
                        45 * random.randint(0, 7)]
            labelOut = os.path.join(outDirFinal, 'test.txt')
            outFile = open(labelOut, 'w')
            imgOut = os.path.join(outDirFinal, 'image')
            if not os.path.isdir(imgOut):
                os.makedirs(imgOut)

            direct = 1
            for id in range(imgNumber):
                x = random.randint(1, 3) * 3
                if startXYZ[1] < -80:
                    direct = 1
                elif startXYZ[1] > 80:
                    direct = -1

                y = direct * int(random.randint(0, 2)) * 3
                xNew = startXYZ[0] + x
                if xNew > 180:
                    xNew -= 360
                if xNew < -177:
                    xNew += 360
                yNew = startXYZ[1] + y
                startXYZ = [xNew, yNew, startXYZ[2]]
                labelAim = " ".join(list(map(str, startXYZ)))
                if labelAim in labels2Name.keys():
                    name = labels2Name[labelAim]
                else:
                    continue
                shutil.copy(os.path.join(imgDir, name), os.path.join(imgOut, name))
                labelFinal = name + " " + " ".join(list(map(str, startXYZ))) + '\n'
                outFile.writelines(labelFinal)
            outFile.close()