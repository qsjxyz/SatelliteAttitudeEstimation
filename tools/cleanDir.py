import os

if __name__ == '__main__':
    dir = '../result'
    for task in os.listdir(dir):
        if task == 'E0' or task=='E1':
            continue
        taskDir = os.path.join(dir, task)
        fileList = os.listdir(taskDir)
        fileList.sort()
        for id, file in enumerate(fileList):
            if file.endswith('.pkl'):
                if id < len(fileList)-1:
                    if fileList[id+1].endswith('.pkl'):
                        os.remove(os.path.join(taskDir, file))
